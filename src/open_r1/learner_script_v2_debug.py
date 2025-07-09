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

import warnings

import torch.utils.data
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_rich_available, is_vllm_available
from trl.models import  unwrap_model_for_generation
from trl.trainer.grpo_trainer import RepeatRandomSampler
from trl.trainer.utils import (
    pad,
    print_prompt_completions_sample,
)

import inspect

if is_wandb_available():
    import wandb

import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments, GPGConfig
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
from open_r1.gpg_trainer import GPGTrainer
from open_r1.utils.data_utils import custom_loading_dataset
from transformers import TrainerCallback
from pathlib import Path
from async_utils import setup_fs_queue, pop_from_fs_queue,SamplerSyncCallback # 新增
from trl.extras.profiling import profiling_decorator, profiling_context
from typing import Any, Callable, Optional, Union
from typing import Dict, Any, Union
import torch.nn.functional as F
logger = logging.getLogger(__name__)


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
# 定义一个修改版的 GPGTrainer，用于学习器
# =================================================================================

class LearnerGPGTrainer(GPGTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 从环境变量获取共享目录路径
        fs_queue_path = os.getenv("FS_QUEUE_PATH",  "/Async/Qwen3-0.6B")

        self.sync_weights_path = Path(
            os.getenv("SYNC_WEIGHTS_PATH", "/tmp/Qwen3-0.6B/gpg_async_weights.pt"))

        # 设置文件队列目录
        self.queue_dir, self.processing_dir = setup_fs_queue(fs_queue_path)

        # 获取当前进程的 rank，用于日志和调试
        self.rank = self.accelerator.process_index

        # 从环境变量或参数中获取超时时间
        # 默认设置为20分钟，以防采样器暂时卡顿
        self.queue_timeout = int(os.getenv("QUEUE_TIMEOUT_SECONDS", "1200"))

        logger.info(
            f"[Rank {self.rank}] Learner initialized. "
            f"Reading from queue: {self.queue_dir}, "
            f"Using processing dir: {self.processing_dir}"
        )




    # def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
    #     mode = "train" if self.model.training else "eval"
    #     metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics
    #
    #     # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
    #     # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
    #     if mode == "eval":
    #         metrics = {f"eval_{key}": val for key, val in metrics.items()}
    #
    #     logs = {**logs, **metrics}
    #     super().log(logs, start_time)
    #     self._metrics[mode].clear()
    #
    #     if self.accelerator.is_main_process and self.log_completions:
    #         if is_rich_available():
    #             print_prompt_completions_sample(
    #                 self._textual_logs["prompt"],
    #                 self._textual_logs["completion"],
    #                 self._textual_logs["rewards"],
    #                 self._textual_logs["advantages"],
    #                 self.state.global_step,
    #                 self.num_completions_to_print,
    #             )
    #
    #         if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
    #             import pandas as pd
    #
    #             table = {
    #                 "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
    #                 "prompt": self._textual_logs["prompt"],
    #                 "completion": self._textual_logs["completion"],
    #                 **self._textual_logs["rewards"],
    #                 "advantage": self._textual_logs["advantages"],
    #             }
    #             torch.save(table, f"/userhome/Research_HUB/GPG/open-r1/wandb/debug/table_line173.pt")
    #             df = pd.DataFrame(table)
    #             if self.wandb_log_unique_prompts:
    #                 df = df.drop_duplicates(subset=["prompt"])
    #             wandb.log({"completions": wandb.Table(dataframe=df)})

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]

        per_token_loss = - per_token_logps * advantages.unsqueeze(1)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        #Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        if self.args.adjust_gd and mode=="train":
            loss = loss / self._metrics[mode]['inverse_alpha'][-1]

        return loss

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        重写 _prepare_inputs 方法。
        此方法将阻塞，直到从共享文件队列中成功获取一个数据包，
        然后将其准备好以用于训练。

        原始的 'inputs' 参数（来自 dummy dataloader）被忽略。
        """
        # 调用 pop_from_fs_queue，它包含了轮询、原子重命名和反序列化逻辑
        # 每个学习器进程都会在这里竞争可用的数据文件
        mode = "train" if self.model.training else "eval"

        if mode == "train":
            rollout_batch = pop_from_fs_queue(
                queue_dir=self.queue_dir,
                processing_dir=self.processing_dir,
                rank=self.rank,
                timeout=self.queue_timeout
            )

            # 处理超时或队列为空的情况
            if rollout_batch is None:
                # 如果 pop_from_fs_queue 返回 None，说明在指定的超时时间内没有获取到任何数据。
                # 这通常意味着采样器进程已经崩溃、卡死或速度远远跟不上学习器。
                # 在这种情况下，我们应该让训练失败，而不是无限期地等待。
                error_message = (
                    f"[Rank {self.rank}] CRITICAL: Timed out after {self.queue_timeout} seconds "
                    f"waiting for data from the file queue at '{self.queue_dir}'. "
                    "The sampler process(es) might be down or stuck. Aborting training."
                )
                logger.error(error_message)
                raise RuntimeError(error_message)

            # 将从文件中加载的数据（目前在 CPU 上）移动到当前进程的 GPU 设备
            try:
                for key, value in rollout_batch.items():
                    if isinstance(value, torch.Tensor):
                        rollout_batch[key] = value.to(self.accelerator.device)
            except Exception as e:
                logger.error(f"[Rank {self.rank}] Failed to move batch to device. Error: {e}")
                # 也可以在这里记录 batch 的 keys，以帮助调试
                logger.error(f"Batch keys: {list(rollout_batch.keys())}")
                raise e

            self._metrics = rollout_batch["metrics"]

            return rollout_batch
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
            return inputs

    def get_train_dataloader(self):
        """
        重写 get_train_dataloader 以创建一个 "dummy" 数据加载器。
        它的唯一作用是驱动训练循环的步数。
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # 创建一个只包含占位符的 IterableDataset
        class DummyInfiniteIterableDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                while True: # 无限迭代
                    yield {}

        return torch.utils.data.DataLoader(
            DummyInfiniteIterableDataset(),
            batch_size=self.args.per_device_train_batch_size,
        )


    def _generate_and_score_completions(
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

        max_gen = 1 if not self.model.training else max_gen # to make validation work
        mode = "train" if self.model.training else "eval"
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
            if self.args.use_vllm:
                # First, have main process load weights if needed
                if self.state.global_step != self._last_loaded_step:
                    self._move_model_to_vllm()
                    self._last_loaded_step = self.state.global_step

                # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

                # Pad the completions, and concatenate them with the prompts
                completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
                completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            else:
                # Regular generation path
                with unwrap_model_for_generation(
                        self.model_wrapped, self.accelerator,
                        gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

                # prompt_completion_ids = pad_sequence_to_length(prompt_completion_ids,
                #                                                self.max_prompt_length+self.max_completion_length,
                #                                                self.processing_class.pad_token_id, left_pad=False)

                # Compute prompt length and extract completion ids
                prompt_length = prompt_ids.size(1)
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

            rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                    zip(self.reward_funcs, self.reward_processing_classes)
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
                        # Convert None values to NaN
                        output_reward_func = [reward if reward is not None else torch.nan for reward in
                                              output_reward_func]

                        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            # If all reward functions return None for a given row, issue a detailed warning
            mode = "train" if self.model.training else "eval"

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
            rewards_per_func = gather(rewards_per_func)

            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

            # calculate data weight based on acc reward
            mean_grouped_rewards_cpu = mean_grouped_rewards.cpu().numpy()
            for i, p in enumerate(ordered_set_of_problems):
                self.data_weight[mode][p].append(float(mean_grouped_rewards_cpu[i]))

            stds = rewards.view(-1, self.num_generations).std(dim=1)
            # 找出标准差为 0 的组
            identical_value_mask = stds == 0
            easy_mask = mean_grouped_rewards == 1
            hard_mask = mean_grouped_rewards == 0

            # 计算标准差为 0 的组的数目
            num_identical_reward_groups = identical_value_mask.sum().item()
            num_easy_problem = easy_mask.sum().item()
            num_hard_problem = hard_mask.sum().item()
            num_samples = stds.numel()

            # 判断是否符合min_inverse_alpha要求，如果不符合，继续取样本；如果符合，进入后续计算。
            n_valid_samples += num_samples - num_identical_reward_groups

            # 每个worker组装自己部分的tensor
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )

            my_rewards = rewards[process_slice] #16

            my_rewards_stds = my_rewards.view(-1, self.num_generations).std(dim=1)#[2]
            my_identical_value_mask = torch.where(my_rewards_stds == 0)[0] #[]
            my_valid_value_mask = torch.where(my_rewards_stds > 0)[0]
            num_questions = len(prompts) // self.num_generations #2
            _b_valid = my_valid_value_mask.shape[0] * self.num_generations
            _b_ident = my_identical_value_mask.shape[0] * self.num_generations
            assert _b_ident + _b_valid == len(prompts) # 16

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

            new_rewards = merge(valid_rewards, new_rewards)

            new_prompt_mask = merge_with_padding(valid_prompt_mask, new_prompt_mask, 0, left_pad=True)
            new_prompt_ids = merge_with_padding(valid_prompt_ids, new_prompt_ids, self.processing_class.pad_token_id, left_pad=True)
            new_completion_mask = merge_with_padding(valid_completion_mask, new_completion_mask, 0, left_pad=False)
            new_completion_ids = merge_with_padding(valid_completion_ids, new_completion_ids, self.processing_class.pad_token_id, left_pad=False)

            if n_valid_samples < self.args.min_inverse_alpha * num_samples and mode == "train": # 加了  and mode == "train"
                logger.info(f"keep generating more examples: the {n_gen}-th mini-batch")
                n_gen += 1

            else:
                # 重新组装样本batch
                merge_rewards = merge(identical_rewards, new_rewards)
                rewards =merge_rewards[:len(prompts)]
                prompt_ids = merge_with_padding(identical_prompt_ids, new_prompt_ids, self.processing_class.pad_token_id, left_pad=True)[:len(prompts)]
                prompt_mask = merge_with_padding(identical_prompt_mask, new_prompt_mask, 0, left_pad=True)[:len(prompts)]
                completion_ids = merge_with_padding(identical_completion_ids, new_completion_ids, self.processing_class.pad_token_id, left_pad=False)[:len(prompts)]
                completion_mask = merge_with_padding(identical_completion_mask, new_completion_mask, 0, left_pad=False)[:len(prompts)]
                break

        if self.model.training:
            assert n_gen < max_gen

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)#[8]
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)#[8]
        g_mean_grouped_rewards = mean_grouped_rewards
        g_std_grouped_rewards = std_grouped_rewards
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)#[8] -> [64]
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards ##[16] -> [64]

        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        inverse_alpha = n_valid_samples / num_samples
        inverse_alpha = min(1.0, inverse_alpha)

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

        # if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
        #     prompts_to_log = gather_object(prompts_text)
        #     completions_to_log = gather_object(completions_text)
        #     rewards_to_log = rewards.tolist()
        #
        #     if self.accelerator.is_main_process:
        #         if is_rich_available():
        #             print_prompt_completions_sample(
        #                 prompts_to_log,
        #                 completions_to_log,
        #                 rewards_to_log,
        #                 self.state.global_step,
        #             )
        #         if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
        #             import pandas as pd
        #
        #             # For logging
        #             table = {
        #                 "step": [str(self.state.global_step)] * len(rewards),
        #                 "prompt": prompts_to_log,
        #                 "completion": completions_to_log,
        #                 "reward": rewards.tolist(),
        #             }
        #
        #             # torch.save(table,f"/userhome/Research_HUB/GPG/open-r1/wandb/debug/table.pt")
        #
        #             df = pd.DataFrame(table)
        #             wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
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
        return {"prompt": prompt}

    def make_conversation_math35(example):
        prompt = []
        # prompt.append({"role": "user", "content": example["instruction"][0]['content']})
        prompt = example["instruction"]
        # prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    if 'simplelr_qwen_level3to5' in script_args.dataset_name:
        dataset = dataset.map(make_conversation_math35)
    else:
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

    #############################
    # Initialize the GRPO trainer
    #############################
    if training_args.eval_strategy == "no":
        eval_dataset = None
    else:
        if training_args.weighted_sample:
            eval_dataset = dataset[script_args.dataset_train_split]
        else:
            eval_dataset = dataset[script_args.dataset_test_split]

    trainer = LearnerGPGTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset.select(range(16)),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    # 添加用于模型同步的回调
    sync_weights_path = Path(os.getenv("SYNC_WEIGHTS_PATH", "/tmp/Qwen3-0.6B/gpg_async_weights.pt"))
    sync_steps = int(os.getenv("SYNC_SAMPLER_STEPS", "1"))

    # 将 trainer 实例自身传递给回调的构造函数
    sync_callback = SamplerSyncCallback(
        trainer=trainer,
        sync_weights_path=sync_weights_path,
        sync_steps=sync_steps
    )
    trainer.add_callback(sync_callback)

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
    logger.info("*** Starting Learner Training Loop ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)





if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GPGConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)


