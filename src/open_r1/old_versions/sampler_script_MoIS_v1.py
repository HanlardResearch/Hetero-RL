# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

readme="""【版本说明】
Version: MIS_v1
功能：采样时保留生成概率
说明：基于 v5修改
"""

from contextlib import nullcontext
import time
import inspect
import logging
import os
import sys
import datasets
import torch
import transformers
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from open_r1.configs import AsyGPGScriptArguments, GPGConfig
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, TrlParser, get_peft_config
from open_r1.gpg_trainer_vllm import GPGTrainer
from open_r1.utils.data_utils import custom_loading_dataset
from transformers import TrainerCallback
from pathlib import Path
from trl.extras.profiling import profiling_decorator, profiling_context
from typing import Any, Union
from async_utils import setup_fs_queue, push_to_fs_queue # 新增

from transformers import set_seed, TrainerControl
import torch.nn.functional as F
import warnings

import torch.utils.data
from accelerate.utils import gather, gather_object, is_peft_model, set_seed

from torch import nn
from transformers import (
    Trainer,
    is_wandb_available,
)
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available
from trl.models import  unwrap_model_for_generation
from trl.trainer.utils import (
    pad,
)
from accelerate.utils import reduce, broadcast
from open_r1.Time_Delay import get_delay_sampler

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb
    # print("wandb has imported")

import torch.distributed as dist

logger = logging.getLogger(__name__)


# def reduce(tensor):
#     return tensor

def merge(valid_rewards, new_rewards):
    if valid_rewards is None:
        return new_rewards
    else:
        if new_rewards is None:
            return valid_rewards
        else:
            return torch.concat([new_rewards, valid_rewards])


def merge_with_padding(valid_rewards, new_rewards, pad_token_id, left_pad=False):
    if valid_rewards is None:
        return new_rewards
    else:
        if new_rewards is None:
            return valid_rewards
        else:
            if new_rewards.shape[1] < valid_rewards.shape[1]:
                new_rewards = pad_sequence_to_length(new_rewards, valid_rewards.shape[1], pad_token_id, left_pad)
            else:
                valid_rewards = pad_sequence_to_length(valid_rewards, new_rewards.shape[1], pad_token_id, left_pad)
            return torch.concat([new_rewards, valid_rewards])


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)




# =================================================================================
# 定义一个专门用于采样的 GPGTrainer
# =================================================================================
class SamplerGPGTrainer(GPGTrainer):
    """
    一个专门用于采样的 GPGTrainer 子类。
    它封装了模型同步、数据生成和与 Redis 通信的所有逻辑。
    """

    def __init__(self,  delay_list, *args, **kwargs):
        # 移除 kwargs 中的 optimizers，以防与我们的虚拟优化器冲突
        self.fs_queue_path = kwargs.pop("fs_queue_path")

        # 使用虚拟优化器安全地初始化父类
        kwargs.pop("optimizers", None)
        super().__init__(*args, optimizers=(None, None), **kwargs)

        # 设置文件队列
        self.queue_dir, _ = setup_fs_queue(self.fs_queue_path)

        self.log_interval = int(os.getenv("SAMPLER_LOG_INTERVAL", "10"))
        self.sync_weights_path = Path(os.getenv("SYNC_WEIGHTS_PATH", "/tmp/Qwen3-0.6B/gpg_async_weights.pt"))
        self.last_sync_time = 0
        self.rank = self.accelerator.process_index
        # 封装内部状态
        self._dataloader = self.get_train_dataloader()
        self._epoch_iterator = iter(self._dataloader)
        self.batch_ids = 0
        self.model_ids = 0
        self.delay_list = delay_list.__iter__()
        logger.info(f"delay_list[:20]: {delay_list[:20]}")

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed
            # logger.info(f"sampler_script_v2_vllm.py line 150")
            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext
            # logger.info(f"sampler_script_v2_vllm.py line 154")
        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                self._sync_fsdp_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
                # logger.info(f"sampler_script_v2_vllm.py line 191")
            else:
                # logger.info(f"sampler_script_v2_vllm.py line 193")
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            # logger.info(f"sampler_script_v2_vllm.py line 206")
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            # logger.info(f"sampler_script_v2_vllm.py line 209")
            self.llm.reset_prefix_cache()

    def generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        max_gen = 20
        n_gen = 1
        n_valid_samples = 0
        device = self.accelerator.device
        new_rewards = None
        new_prompt_ids = None
        new_prompt_mask = None
        new_completion_ids = None
        new_completion_mask = None
        mode = "train"  # 采样的数据用于训练  所以是 train

        max_gen = 1 if mode!= "train" else max_gen # to make validation work

        while n_gen <= max_gen:
            if n_gen == 1 and len(inputs) > 0: # the dataloader iter finishes, we need a new iter.
                inputs = inputs
            else:
                epoch_iterator = self._epoch_iterator
                batch_samples, num_items_in_batch, end = self.get_local_batch_samples(epoch_iterator, 1)
                if end: # reset dataloader since this epoch doesn't end.
                    frame = inspect.currentframe().f_back.f_back.f_back
                    # logger.info("frame keys ", frame.f_locals.keys())
                    # 查找 epoch_iterator
                    epoch_dataloader = frame.f_locals.get('epoch_dataloader')
                    epoch_iterator = iter(epoch_dataloader)
                    frame.f_locals['epoch_iterator'] = epoch_iterator
                    self._epoch_iterator = epoch_iterator
                    batch_samples, num_items_in_batch, end = self.get_local_batch_samples(epoch_iterator, 1)
                inputs = batch_samples[0]
            prompts = [x["prompt"] for x in inputs]
            if isinstance(inputs[0]["prompt"], (list,)):
                problems = [x["prompt"][-1]['content'] for x in inputs]
            else:
                problems = prompts[:]
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            prompt_inputs = self.processing_class(
                text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length:]
                prompt_mask = prompt_mask[:, -self.max_prompt_length:]
                # prompt_ids = pad_sequence_to_length(prompt_ids, self.max_prompt_length, self.processing_class.pad_token_id, left_pad=True)
                # prompt_mask = pad_sequence_to_length(prompt_mask, self.max_prompt_length, 0, left_pad=True)

            # Generate completions using either vLLM or regular generation
            time_start = time.time()
            if self.args.use_vllm:
                # First, have main process load weights if needed
                if self.state.global_step != self._last_loaded_step:
                    self._move_model_to_vllm()
                    self._last_loaded_step = self.state.global_step

                # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text
                # print(f'>>>>>>>>>>>>generation_kwargs:{generation_kwargs}')
                # print(f'>>>>>>>>>>>>self.args.generation_kwargs:{self.args.generation_kwargs}')
                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]
                # Pad the completions, and concatenate them with the prompts
                completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
                completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            else:
                # Regular generation path
                #print(f'>>>>>>>>>>>>prompt_ids:{prompt_ids}')
                #print(f'>>>>>>>>>>>>attention_mask:{prompt_mask}')
                #print(f'>>>>>>>>>>>>generation_config:{self.generation_config}')
                # print(f'>>>>>>>>>>>>self.model_args:{self.model_args}')

                with unwrap_model_for_generation(
                        self.model_wrapped, self.accelerator,
                        gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )
                
                #print(f'>>>>>>>>>>>>prompt_completion_ids:{prompt_completion_ids}')
                #print(f'prompt_completion_ids.shape:{prompt_completion_ids.shape}')
                # prompt_completion_ids = pad_sequence_to_length(prompt_completion_ids,
                #                                                self.max_prompt_length+self.max_completion_length,
                #                                                self.processing_class.pad_token_id, left_pad=False)

                # Compute prompt length and extract completion ids
            time_end = time.time()
            # logger.info(f"prompt_completion_ids生成时长： {(time_end - time_start):.2f}s")
            prompt_length = prompt_ids.size(1)
            # logger.info(f"current rank:{self.rank}")
            # logger.info(f"prompt_ids.shape:{prompt_ids.shape}")
            # logger.info(f"prompt_ids:{prompt_ids}")
            # logger.info(f"prompt_completion_ids.shape:{prompt_completion_ids.shape}")
            # logger.info(f"prompt_completion_ids:{prompt_completion_ids}")
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            # all_prompts_text = gather_object(prompts_text)
            all_problems = gather_object(problems)
            ordered_set_of_problems = all_problems[:: self.num_generations]
            # ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            # Concatenate prompt_mask with completion_mask for logit computation
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

            logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

            with torch.no_grad():
                # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
                # computation here, and use per_token_logps.detach() instead.
                if self.num_iterations > 1:
                    old_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    old_per_token_logps = None

                if self.beta == 0.0:
                    ref_per_token_logps = None
                elif self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )



            # Decode the generated completions
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                completions = []
                for prompt, completion in zip(prompts, completions_text):
                    bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                    completions.append([{"role": "assistant", "content": bootstrap + completion}])
            else:
                completions = completions_text
            # ['accuracy_lv35', 'tag_count', 'length', 'repetition_penalty']
            rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class, RF_name) in enumerate(
                    zip(self.reward_funcs, self.reward_processing_classes,script_args.reward_funcs)
            ):
                if isinstance(reward_func,
                              nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                    reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                else:
                    reward_func_name = reward_func.__name__
                with profiling_context(self, reward_func_name):
                    if isinstance(
                            reward_func, nn.Module
                    ):  # Module instead of PretrainedModel for compat with compiled models
                        if is_conversational(inputs[0]):
                            messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                            texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                        else:
                            texts = [p + c for p, c in zip(prompts, completions)]
                        reward_inputs = reward_processing_class(
                            text=texts, return_tensors="pt", padding=True, padding_side="right",
                            add_special_tokens=False
                        )
                        reward_inputs = super()._prepare_inputs(reward_inputs)
                        with torch.inference_mode():
                            rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                    else:
                        # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                        keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                        output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                        # logger.info("*"*50+"Q:")
                        # logger.info(f"prompts:{prompts}")
                        # logger.info("*"*50+"A:")
                        # logger.info(f"completions:{completions}")
                        # logger.info("*"*50+"Reward")
                        logger.info(f"{RF_name}:{output_reward_func}")
                        # logger.info("*"*50)
                        # Convert None values to NaN
                        output_reward_func = [reward if reward is not None else torch.nan for reward in
                                              output_reward_func]

                        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            # If all reward functions return None for a given row, issue a detailed warning


            if torch.isnan(rewards_per_func).all(dim=1).any():
                nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
                row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
                row_reward_kwargs["prompt"] = prompts[nan_row_idx]
                row_reward_kwargs["completion"] = completions[nan_row_idx]
                warnings.warn(
                    f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                    "Please ensure that at least one reward function returns a valid reward."
                )

            # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
            # completions may be distributed across processes
            rewards_per_func = gather(rewards_per_func) # [32,]

            ########################################## 使用acc_reward 计算方差 ################################################
            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)# [32,]
            acc_rewards = rewards_per_func[:,0] # 第一个奖励函数是 'accuracy_lv35'

            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # [4,1]
            mean_grouped_acc_rewards = acc_rewards.view(-1, self.num_generations).mean(dim=1) #备用 # [4,1]

            # calculate data weight based on acc reward
            mean_grouped_rewards_cpu = mean_grouped_rewards.cpu().numpy()

            for i, p in enumerate(ordered_set_of_problems):
                self.data_weight[mode][p].append(float(mean_grouped_rewards_cpu[i]))

            stds = rewards.view(-1, self.num_generations).std(dim=1) # [4,1]
            acc_stds = acc_rewards.view(-1, self.num_generations).std(dim=1) #备用 # [4,1]

            # 找出标准差为 0 的组 (根据acc_reward)
            # identical_value_mask = stds == 0 #
            identical_value_mask = acc_stds == 0 # [4,1]
            easy_mask = mean_grouped_acc_rewards == 1 # [4,1]
            hard_mask = mean_grouped_acc_rewards == 0 # [4,1]

            # 计算标准差为 0 的组的数目
            num_identical_reward_groups = identical_value_mask.sum().item() #[1,]
            num_easy_problem = easy_mask.sum().item() #[1,]
            num_hard_problem = hard_mask.sum().item() #[1,]
            num_samples = stds.numel() #[1,] :=4

            # 判断是否符合min_inverse_alpha要求，如果不符合，继续取样本；如果符合，进入后续计算。
            n_valid_samples += num_samples - num_identical_reward_groups # :~ [0,4]

            # 每个worker组装自己部分的tensor
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            ) #[8,]
            my_rewards = rewards[process_slice] #[8,]
            my_acc_rewards = acc_rewards[process_slice] #[8,]

            # my_rewards_stds = my_rewards.view(-1, self.num_generations).std(dim=1)
            my_acc_rewards_stds = my_acc_rewards.view(-1, self.num_generations).std(dim=1) #[1,]

            # 这个mask必须对应于acc_reward
            my_identical_value_mask = torch.where(my_acc_rewards_stds == 0)[0] #[1,]
            my_valid_value_mask = torch.where(my_acc_rewards_stds > 0)[0] #[1,]

            num_questions = len(prompts) // self.num_generations #[1,] :=1
            _b_valid = my_valid_value_mask.shape[0] * self.num_generations # [1,]:= 0 or 8
            _b_ident = my_identical_value_mask.shape[0] * self.num_generations # [1,]:= 0 or 8
            assert _b_ident + _b_valid == len(prompts)
            ########################################## 使用acc_reward 计算方差 ################################################


            if _b_valid > 0:
                valid_rewards = my_rewards.reshape(num_questions, self.num_generations)[my_valid_value_mask].reshape(_b_valid)
                valid_prompt_ids = prompt_ids.reshape(num_questions, self.num_generations, -1)[my_valid_value_mask].reshape(_b_valid, -1)
                valid_prompt_mask = prompt_mask.reshape(num_questions, self.num_generations, -1)[my_valid_value_mask].reshape(_b_valid, -1)
                valid_completion_ids = completion_ids.reshape(num_questions, self.num_generations, -1)[
                    my_valid_value_mask].reshape(_b_valid, -1)
                valid_completion_mask = completion_mask.reshape(num_questions, self.num_generations, -1)[
                    my_valid_value_mask].reshape(_b_valid, -1)
            else:
                valid_rewards, valid_prompt_ids, valid_prompt_mask, valid_completion_ids, valid_completion_mask = [None] * 5
            if _b_ident > 0:
                identical_rewards = my_rewards.reshape(num_questions, self.num_generations)[
                    my_identical_value_mask].reshape(_b_ident)
                identical_prompt_ids = prompt_ids.reshape(num_questions, self.num_generations, -1)[
                    my_identical_value_mask].reshape(_b_ident, -1)
                identical_prompt_mask = prompt_mask.reshape(num_questions, self.num_generations, -1)[
                    my_identical_value_mask].reshape(_b_ident, -1)
                identical_completion_ids = completion_ids.reshape(num_questions, self.num_generations, -1)[
                    my_identical_value_mask].reshape(_b_ident, -1)
                identical_completion_mask = completion_mask.reshape(num_questions, self.num_generations, -1)[
                    my_identical_value_mask].reshape(_b_ident, -1)
            else:
                identical_rewards, identical_prompt_ids, identical_prompt_mask, identical_completion_ids, identical_completion_mask = [None] * 5

            new_rewards = merge(valid_rewards, new_rewards)# always 有效
            new_prompt_mask = merge_with_padding(valid_prompt_mask, new_prompt_mask, 0, left_pad=True)
            new_prompt_ids = merge_with_padding(valid_prompt_ids, new_prompt_ids, self.processing_class.pad_token_id, left_pad=True)
            new_completion_mask = merge_with_padding(valid_completion_mask, new_completion_mask, 0, left_pad=False)
            new_completion_ids = merge_with_padding(valid_completion_ids, new_completion_ids, self.processing_class.pad_token_id, left_pad=False)

            if n_valid_samples < self.args.min_inverse_alpha * num_samples:
                logger.info(f"keep generating more examples: the {n_gen}-th mini-batch, n_valid_samples:{n_valid_samples}<{self.args.min_inverse_alpha} * {num_samples} ")
                # 这里可以考虑改变超参数
                n_gen += 1

            else:
                # 重新组装样本batch
                merge_rewards = merge(identical_rewards, new_rewards)
                rewards =merge_rewards[:len(prompts)] # [8,]
                prompt_ids = merge_with_padding(identical_prompt_ids, new_prompt_ids, self.processing_class.pad_token_id, left_pad=True)[:len(prompts)]
                prompt_mask = merge_with_padding(identical_prompt_mask, new_prompt_mask, 0, left_pad=True)[:len(prompts)]
                completion_ids = merge_with_padding(identical_completion_ids, new_completion_ids, self.processing_class.pad_token_id, left_pad=False)[:len(prompts)]
                completion_mask = merge_with_padding(identical_completion_mask, new_completion_mask, 0, left_pad=False)[:len(prompts)]
                break


        if mode=="train":
            assert n_gen < max_gen

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # [1, ]
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # [1, ]
        g_mean_grouped_rewards = mean_grouped_rewards
        g_std_grouped_rewards = std_grouped_rewards
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards # [8, ]
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        inverse_alpha = n_valid_samples / num_samples
        inverse_alpha = min(1.0, inverse_alpha)


        ################# 记录采样器的生成概率 #####################
        with torch.no_grad():
            sampler_per_token_logps = self._get_per_token_logps(
                self.model,
                input_ids=torch.cat([prompt_ids, completion_ids], dim=1),
                attention_mask=torch.cat([prompt_mask, completion_mask], dim=1),  # (B, P+C)
                logits_to_keep=completion_ids.size(1),
            )
        ################# 记录采样器的生成概率 #####################

        # Log the metrics
        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(g_mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(g_std_grouped_rewards.mean().item())
        self._metrics[mode]['num_identical_reward_groups'].append(num_identical_reward_groups)
        self._metrics[mode]['num_samples'].append(num_samples)
        self._metrics[mode]['inverse_alpha'].append(inverse_alpha)
        self._metrics[mode]['num_easy_problem'].append(num_easy_problem)
        self._metrics[mode]['num_hard_problem'].append(num_hard_problem)
        # self._metrics[mode]['model_ids'].append(self.model_ids)
        self._metrics[mode]['model_ids'] = [self.model_ids]

        # print(f" wandb.run is not None={ wandb.run is not None} self.batch_ids={self.batch_ids}, self.log_completions={self.log_completions}, self.accelerator.is_main_process={self.accelerator.is_main_process}")


        if self.log_completions:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards = gather(rewards)
            if self.accelerator.is_main_process:
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "model_sync_times": [str(self.model_ids)]* len(rewards),
                        "step": [str(self.batch_ids)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }

                    # print(f"table is done (sampler_script_v2.py)")
                    # torch.save(table,f"/userhome/Research_HUB/GPG/open-r1/wandb/debug/table.pt")

                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        self.batch_ids+=1

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "sampler_per_token_logps":sampler_per_token_logps,
            "advantages": advantages,
            "metrics": self._metrics,
        }


    def _sync_model_weights(self):
        """检查并加载学习器分享的最新模型权重。"""
        if not self.sync_weights_path.exists():
            return

        try:
            current_mtime = self.sync_weights_path.stat().st_mtime
            if current_mtime > self.last_sync_time:
                logger.info(f"[Sampler Rank-{self.rank}] Detected new weights from {self.sync_weights_path}. Loading...")

                global_step, state_dict = torch.load(self.sync_weights_path, map_location="cpu") # d20250717修改

                self.model.load_state_dict(state_dict)
                self._move_model_to_vllm()
                self.last_sync_time = current_mtime
                old_ids = self.model_ids
                self.model_ids = global_step # d20250717修改
                logger.info(f"[Sampler Rank-{self.rank}] New weights loaded successfully. model_ids:{old_ids}->{self.model_ids}")

        except FileNotFoundError:
            # 文件可能在检查存在性和获取 stat 之间被删除，忽略即可
            pass
        except Exception as e:
            logger.error(f"[Sampler] Rank-{self.rank} Error loading weights: {e}")
            # 等待一小会儿，防止因文件正在写入而出错时频繁重试
            time.sleep(1)

    def _get_next_batch(self):
        """封装获取下一批数据的逻辑，包括自动重置数据迭代器。"""
        try:
            return next(self._epoch_iterator)
        except StopIteration:
            logger.info(f"[Sampler] Rank-{self.rank} Dataset depleted. Re-creating dataloader iterator.")
            self._epoch_iterator = iter(self._dataloader)
            return next(self._epoch_iterator)

    def run_sampling_loop(self):
        """
        启动并运行主采样循环。
        这个循环会持续生成数据，直到进程被外部终止。
        """

        logger.info(f"*** Starting Sampler Loop (PID: {os.getpid()}) on device: {self.accelerator.device} ***")
        batch_counter = 0

        delay_time = broadcast(torch.tensor([next(self.delay_list)], device=self.accelerator.device, dtype=torch.float64),from_process=0)

        logger.info(f"first_delay_time:{delay_time}")
        last_time = broadcast(torch.tensor([time.time()], device=self.accelerator.device, dtype=torch.float64),from_process=0) # 记录上一次同步时间
        logger.info(f"first_last_time:{last_time}")
        while True:
            # 1. 同步模型
            now_time =  broadcast(torch.tensor([time.time()], device=self.accelerator.device, dtype=torch.float64),from_process=0) # 记录上一次同步时间
            wait_time =now_time - last_time
            if wait_time > delay_time:
                self._sync_model_weights()
                print(f"[RANK-{self.rank}] 同步模型完成,延迟 {wait_time.item():.1f}(>{delay_time.item():.1f}) 秒")
                last_time =  broadcast(torch.tensor([time.time()], device=self.accelerator.device, dtype=torch.float64),from_process=0) # 更新上次同步时间
                delay_time = broadcast(torch.tensor([next(self.delay_list)], device=self.accelerator.device, dtype=torch.float64),from_process=0) # 采样同步时间延迟
            else:
                print(f"[RANK-{self.rank}] 模型传输进度: {(wait_time/delay_time*100).item():.1f}%, 耗时: {wait_time.item():.1f}(<{delay_time.item():.1f})秒")

            # 2. 获取下一批 prompt
            batch = self._get_next_batch()
            # logger.info(f"# 2. 获取下一批 prompt 完成")

            # 3. 生成和评分
            with torch.no_grad():
                # GPGTrainer 的内部逻辑可能依赖 self.control
                self.control = TrainerControl()
                # time_start = time.time()
                rollout_data = self.generate_and_score_completions(batch)
                # time_end = time.time()
                # logger.info(f"rollout_data生成时长： {time_end - time_start}")
            # logger.info(f"# 3. 生成和评分 完成")
            
            time_save = reduce(torch.tensor([time.time()],device=self.accelerator.device, dtype=torch.float64))
            # 4. 将结果写入文件队列
            cpu_rollout_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in rollout_data.items()}
            push_to_fs_queue(self.queue_dir, cpu_rollout_data, time_save, self.rank)
            # logger.info(f"# 4. 将结果写入文件队列 完成")

            # 5. 日志记录
            batch_counter += 1
            if batch_counter % self.log_interval == 0:
                queue_size = len(list(self.queue_dir.glob("data_*.pt")))
                logger.info(f"[RANK-{self.rank}] Generated and wrote batch #{batch_counter}. Approximate queue size: {queue_size} ")
                logger.info(f"字段： {rollout_data.keys()}")
            # logger.info(f"# 5. 日志记录 完成")



# =================================================================================
# 主函数
# =================================================================================

def main(script_args, training_args, model_args):
    # 设置固定的随机种子，用于延迟时间采样
    rank = training_args.local_rank
    delay_sampler=get_delay_sampler(script_args)
    delay_list = delay_sampler.get_delay_list(n=50000)
    print(f"[RANK-{rank}] delay_list[:20]: {delay_list[:20]}")

    # 设置随时间变化的随机种子，用于多脚本数据采样
    rank =training_args.local_rank
    # seed= int((rank+1)*(int(time.time()*1000)%365))
    seed = int(rank+1)*training_args.seed
    training_args.seed = seed
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"[Rank={rank}] Random seed {seed}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)
        if rank==0:
            current_file_path = __file__
            current_file_name = os.path.basename(current_file_path)
            wandb.login()
            wandb.init(project=os.environ["WANDB_PROJECT"],
                       entity = os.environ["WANDB_ENTITY"],
                       # config=dict(training_args),
                       name=current_file_name,
                       )

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)


    # handle dataset
    # Load the dataset
    if 'simplelr_qwen_level3to5' in script_args.dataset_name:
        dataset = custom_loading_dataset(script_args.dataset_name, max_length=training_args.max_prompt_length, tokenizer=tokenizer)

    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Format into conversation
    def make_conversation(example):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt.append({"role": "user", "content": example["problem"]})


        # prompt = example["problem"] + " The reasoning process MUST BE enclosed within <think> and </think> tags. Please reason step by step, and put your final answer within \\boxed{}."
        # if add_think:
        #     prompt += " /think"

        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    fs_queue_path = os.getenv("FS_QUEUE_PATH", "/extrahome0/save_dir/GPG/4gpus/Async/Qwen3-0.6B")



    trainer = SamplerGPGTrainer(
        delay_list=delay_list,
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
        fs_queue_path=fs_queue_path,

    )

    class ResetDataLoader(TrainerCallback):
        trainer = None

        def on_epoch_end(self, args, state, control, **kwargs):
            """
            Event called at the end of an epoch.
            """
            if hasattr(self.trainer, '_epoch_iterator'):
                print('reset epoch iter in trainer')
                del self.trainer._epoch_iterator

    ResetDataLoader.trainer = trainer
    trainer.add_callback(ResetDataLoader)

    ###############
    # Training loop
    ###############
    logger.info("*** Starting Sampler Sampling Loop ***")
    # 启动采样循环
    trainer.run_sampling_loop()




if __name__ == "__main__":
    parser = TrlParser((AsyGPGScriptArguments, GPGConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    main(script_args, training_args, model_args)


