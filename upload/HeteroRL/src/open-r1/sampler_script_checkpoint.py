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
Version: v5
退火重要性采样时，保留sampler 概率
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
from open_r1.configs import HeteroRLScriptArguments, GRPOConfig
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
from open_r1.utils.data_utils import custom_loading_dataset
from transformers import TrainerCallback
from pathlib import Path
from trl.extras.profiling import profiling_decorator, profiling_context
from typing import Any, Union
from async_utils_checkpoint import setup_fs_queue, push_to_fs_queue # 新增

from transformers import TrainerControl
import torch.nn.functional as F
import warnings
import copy

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
import re
from trl.trainer.utils import selective_log_softmax
from torch.nn.utils.rnn import pad_sequence
import json
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

# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


class SamplerGRPOTrainer(GRPOTrainer):


    def __init__(self,  delay_list, online_mode, *args, **kwargs):

        self.fs_queue_path = kwargs.pop("fs_queue_path")

        kwargs.pop("optimizers", None)

        super().__init__(*args, optimizers=(None, None), **kwargs)

        self.rank = self.accelerator.process_index

        logger.info(
            f"[Rank {self.rank}] grpo sampler initialized. "
        )

        self.queue_dir, _ = setup_fs_queue(self.fs_queue_path)

        self.log_interval = int(os.getenv("SAMPLER_LOG_INTERVAL", "10"))
        self.sync_weights_path = Path(os.getenv("SYNC_WEIGHTS_PATH", "./async_weights.pt"))
        self.last_sync_time = 0

        self._dataloader = self.get_train_dataloader()
        self._epoch_iterator = iter(self._dataloader)
        self.batch_ids = 0
        self.model_ids = 0
        if "checkpoint" in self.model.config._name_or_path:
            self.model_ids = int(self.model.config._name_or_path.split("checkpoint-")[-1])

        self.delay_list = delay_list.__iter__()
        self.online_mode = online_mode
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

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def new_get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        # prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [
                re.sub(rf"^({re.escape(self.processing_class.pad_token)})+", "", text) for text in prompts_text
            ]

        # Generate completions using either vLLM or regular generation
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
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

            sampler_per_token_logps = self._get_per_token_logps(
                self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
            ) if self.loss_type in ["grpo","bnpo", "dr_grpo", "gspo", "gepo", "EqP"] else None
            
            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )
            else:
                ref_per_token_logps = None
        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            
        if self.loss_type == "bnpo":
            min_value, max_value=0.0, 1.0
            rewards = (rewards - min_value)/(max_value - min_value)
            batch_mean = mean_grouped_rewards.mean()
            batch_var = mean_grouped_rewards.std()
            a = (batch_mean*(1-batch_mean)/batch_var-1)*batch_mean if batch_var > 0 else torch.tensor(0.0)
            b = (batch_mean*(1-batch_mean)/batch_var-1)*(1-batch_mean) if batch_var > 0 else torch.tensor(0.0)
            alpha = torch.clamp(1+a/3, min=1.0)
            beta = torch.clamp(1+b/3, min=1.0)
            weight = torch.distributions.Beta(alpha, beta).log_prob(mean_grouped_rewards).exp()
            weight = torch.clamp(1/weight, min=0, max=1e6)
            advantages = weight * (rewards - mean_grouped_rewards)
        else:
            # Normalize the rewards to compute the advantages
            advantages = rewards - mean_grouped_rewards
            if self.loss_type in ["grpo", "gspo"] or self.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]


        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]['model_ids'] = [self.model_ids]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        if self.log_completions:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
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



                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        self.batch_ids+=1
        self_metrics = copy.deepcopy(self._metrics)
        self._metrics[mode].clear()
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "model_ids": self.model_ids,  # 为了根据延迟计算lambdaT
            "sampler_per_token_logps": sampler_per_token_logps,
            "metrics": self_metrics,
        }
    
    @profiling_decorator
    def _sync_model_weights(self, current_idx, file_path, timeout: int = 600):
        """检查并加载学习器分享的最新模型权重。"""
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:  
                if not self.sync_weights_path.exists():
                    if self.online_mode:
                        time.sleep(0.1)
                        continue
                    else:
                        return          
                current_mtime = self.sync_weights_path.stat().st_mtime
                if current_mtime > self.last_sync_time:
                    logger.info(f"[Sampler Rank-{self.rank}] Detected new weights from {self.sync_weights_path}. Loading...")

                    should_update, global_step, state_dict = torch.load(self.sync_weights_path, map_location="cpu") # d20250717修改
                    if should_update:
                        with open(file_path, 'w') as f:
                            json.dump(current_idx, f)
                        print(f"save current index of training dataset: {current_idx}")

                    self.model.load_state_dict(state_dict)
                    self._move_model_to_vllm()
                    self.last_sync_time = current_mtime
                    old_ids = self.model_ids
                    self.model_ids = global_step # d20250717修改
                    logger.info(f"[Sampler Rank-{self.rank}] New weights loaded successfully. model_ids:{old_ids}->{self.model_ids}")
                    return
                # In online mode, sampler will keep waiting for the latest model weight
                if self.online_mode:
                    time.sleep(0.1)
                    continue
                else:
                    logger.info(f"[Sampler Rank-{self.rank}] Weights have not been updated yet. Keep using old weights for sampling.")
                    return
            print(f"WARNING [Rank {self.rank}]: Timed out after {time.time() - start_time} (Max-{timeout})s waiting for model weight from '{self.sync_weights_path}'.")
            error_message = (
                f"[Rank {self.rank}] CRITICAL: Timed out after {timeout} seconds "
                f"waiting for model weight from '{self.sync_weights_path}'. "
                "The learner process(es) might be down or stuck. Aborting training."
            )
            logger.error(error_message)
            raise RuntimeError(error_message)
        except FileNotFoundError:
            # 文件可能在检查存在性和获取 stat 之间被删除，忽略即可
            pass
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"[Sampler] Rank-{self.rank} Error loading weights: {e}")
            # 等待一小会儿，防止因文件正在写入而出错时频繁重试
            time.sleep(1)

    def _get_next_batch(self, index):
        """封装获取下一批数据的逻辑，包括自动重置数据迭代器。"""
        try:
            index += 1
            return next(self._epoch_iterator), index
        except StopIteration:
            logger.info(f"[Sampler] Rank-{self.rank} Dataset depleted. Re-creating dataloader iterator.")
            self._epoch_iterator = iter(self._dataloader)
            index = 0
            return next(self._epoch_iterator), index

    def run_sampling_loop(self, file_path):
        """
        启动并运行主采样循环。
        这个循环会持续生成数据，直到进程被外部终止。
        """

        logger.info(f"*** Starting Sampler Loop (PID: {os.getpid()}) on device: {self.accelerator.device} ***")
        batch_counter = 0

        delay_time = broadcast(torch.tensor([next(self.delay_list)], device=self.accelerator.device, dtype=torch.float64),from_process=0)

        logger.info(f"first_delay_time:{delay_time}")
        last_time = broadcast(torch.tensor([time.time()], device=self.accelerator.device, dtype=torch.float64),from_process=0)
        logger.info(f"first_last_time:{last_time}")
        current_index = -1

    
        while True:

            now_time =  broadcast(torch.tensor([time.time()], device=self.accelerator.device, dtype=torch.float64),from_process=0)
            wait_time =now_time - last_time
            if wait_time > delay_time:
                self._sync_model_weights(current_index, file_path)
                print(f"[RANK-{self.rank}] model sync,latency {wait_time.item():.1f}(>{delay_time.item():.1f}) s")
                last_time =  broadcast(torch.tensor([time.time()], device=self.accelerator.device, dtype=torch.float64),from_process=0)
                delay_time = broadcast(torch.tensor([next(self.delay_list)], device=self.accelerator.device, dtype=torch.float64),from_process=0)
            else:
                print(f"[RANK-{self.rank}] : {(wait_time/delay_time*100).item():.1f}%, spend time: {wait_time.item():.1f}(<{delay_time.item():.1f}) s")


            batch, current_index = self._get_next_batch(current_index)


            with torch.no_grad():
                self.control = TrainerControl()
                rollout_data = self.generate_and_score_completions(batch)

            
            time_save = reduce(torch.tensor([time.time()],device=self.accelerator.device, dtype=torch.float64))

            cpu_rollout_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in rollout_data.items()}
            push_to_fs_queue(self, cpu_rollout_data, time_save)


            batch_counter += 1
            if batch_counter % self.log_interval == 0:
                queue_size = len(list(self.queue_dir.glob("data_*.pt")))
                logger.info(f"[RANK-{self.rank}] Generated and wrote batch #{batch_counter}. Approximate queue size: {queue_size} ")





def main(script_args, training_args, model_args):

    rank = training_args.local_rank
    delay_sampler=get_delay_sampler(script_args)
    delay_list = delay_sampler.get_delay_list(n=50000)
    print(f"[RANK-{rank}] delay_list[:20]: {delay_list[:20]}")

    rank =training_args.local_rank
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
    logger.info(f"[Rank={rank}] Random seed {training_args.seed}")
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
                       name=current_file_name
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
        # if training_args.system_prompt is not None:
        #     prompt.append({"role": "system", "content": training_args.system_prompt})
        if script_args.use_think:
            prompt.append({"role": "system", "content": script_args.system_prompt_think})
            prompt.append({"role": "user", "content": example["problem"]})
        else:
            prompt.append({"role": "system", "content": script_args.system_prompt_nothink})
            prompt.append({"role": "user", "content": example["problem"] + "/no_think"})

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

    fs_queue_path = os.getenv("FS_QUEUE_PATH", "./")

    if script_args.num_samplers != 1:
        shuffled_dataset = dataset[script_args.dataset_train_split].shuffle(seed=training_args.seed)
        sampling_dataset = shuffled_dataset.shard(num_shards=script_args.num_samplers, index=script_args.sampler_id)
    else:
        sampling_dataset = dataset[script_args.dataset_train_split]
    
    # file_path = os.path.join(training_args.output_dir, 'last_index.json')
    file_path = Path(training_args.output_dir) / 'last_index.json'
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint == "True":
            if file_path.exists():
                with open(file_path, 'r') as f:
                    last_index = json.load(f)
                start_idx = last_index + 1
                logger.info(f"Start training from existing dataset index {start_idx}")
                if start_idx >= len(sampling_dataset):
                    logger.info("training complete!")
                    exit()
                sampling_dataset = sampling_dataset.select(range(start_idx, len(sampling_dataset)))
            else:
                logger.info(f"No index file found! Start from scratch.")
    
    trainer = SamplerGRPOTrainer(
        delay_list=delay_list,
        online_mode=script_args.online_mode,
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=sampling_dataset,
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
    trainer.run_sampling_loop(file_path)

if __name__ == "__main__":
    parser = TrlParser((HeteroRLScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    main(script_args, training_args, model_args)


