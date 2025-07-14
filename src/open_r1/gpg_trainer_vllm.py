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
import logging
logger = logging.getLogger(__name__)

import os
import textwrap
import warnings
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch
import random
import torch
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
from transformers.utils import is_rich_available, is_peft_available
from transformers.trainer_callback import ExportableState
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_decorator, profiling_context
from trl.import_utils import is_vllm_available
from trl.models import  unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import RepeatSampler
from trl.trainer.utils import (
    pad,
    print_prompt_completions_sample,
)
from packaging import version
import transformers
from collections import defaultdict
import inspect

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
from torch.utils.data import Sampler
import torch.nn.functional as F


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class AdpativeRepeatRandomSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
        trainer = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)
        self.trainer = trainer

    def __iter__(self):
        if len(self.trainer.data_weight['train']) < 1 and len(self.trainer.data_weight['eval']) < 1:
            # # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

            #    [2, 4, 3, 1, 0, 6, 5]
            # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
            indexes = [indexes[i: i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

            #    [[2, 4, 3], [1, 0, 6], [5]]
            # -> [[2, 4, 3], [1, 0, 6]]
            indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]
        else:
            def weight_soft(rewards):
                if len(rewards) < 1:
                    return 1.0
                else:
                    mean = sum(rewards) / len(rewards)
                    if mean == 1.0:
                        return 0.0
                    elif mean == 0.0:
                        return 0.0
                    else:
                        return 1.0 - mean

            def weight_uniform(rewards):
                return 1.0

            def weight_hard(rewards):
                # more hard problem to be solved
                if len(rewards) < 1:
                    return 1.0
                else:
                    mean = sum(rewards) / len(rewards)
                    return 1 - mean

            def weight_medium(rewards):
                if len(rewards) < 1:
                    return 1.0
                else:
                    mean = sum(rewards) / len(rewards)
                    if mean == 0 or mean == 1:
                        return 0
                    else:
                        return 1

            strategy = {'uniform': weight_uniform, 'hard': weight_hard, 'soft': weight_soft, 'medium': weight_medium}
            weight_cal = strategy[self.trainer.args.sample_strategy]
            data_weight = self.trainer.data_weight['eval'] if len(self.trainer.data_weight['eval']) > 0 else self.trainer.data_weight['train']

            # assert len(self.data_source[0]['problem']) == 1
            if isinstance(self.data_source[0]["prompt"], (dict,)):
                prompts = [self.data_source[i]["prompt"][-1]['content'] for i in range(self.num_samples)]
            else:
                prompts = [self.data_source[i]["prompt"] for i in range(self.num_samples)]

            weights = [weight_cal(data_weight[prompt]) for prompt in prompts]
            indexes = list(range(self.num_samples))
            sampled_indexes = random.choices(indexes, weights=weights, k=self.num_samples)

            # 将采样后的索引分批
            sampled_indexes = [sampled_indexes[i: i + self.batch_size] for i in
                               range(0, len(sampled_indexes), self.batch_size)]
            # 过滤不完整的批次
            sampled_indexes = [chunk for chunk in sampled_indexes if len(chunk) == self.batch_size]
            indexes = sampled_indexes

            # 更新 data_weight：被选中的样本权重清空
            for chunk in sampled_indexes:
                for index in chunk:
                    prompt = prompts[index]
                    if len(data_weight[prompt]) > 0:
                        del data_weight[prompt]  # 清空被选中的样本权重

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class GPGTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # args.vllm_mode = "colocate"
        super().__init__(model, reward_funcs, args, train_dataset, eval_dataset, processing_class, reward_processing_classes, callbacks,
                         optimizers,peft_config)
        self.scale_batch = args.scale_batch
        self.data_weight = {'train': defaultdict(list), 'eval': defaultdict(list)}

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        effective_batch_size = (
            self.args.per_device_eval_batch_size
            * self.accelerator.num_processes
        )
        # num_generations = self.num_generations // 2
        num_generations = self.num_generations
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=num_generations,  # save cost.
            batch_size= effective_batch_size // num_generations,
            seed=self.args.seed,
        )

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                     |     GPU 0     |     GPU 1     |     GPU 2    |
        #
        #               global_step   step     <───────>  num_generations=3
        #                                      <───────────> per_device_train_batch_size=4
        #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │
        #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
        #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
        #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
        #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15
        #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19
        #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        if self.args.weighted_sample:
            return AdpativeRepeatRandomSampler(
                data_source=self.train_dataset,
                mini_repeat_count=self.num_generations,
                batch_size=effective_batch_size // self.num_generations,
                repeat_count=self.num_iterations,
                seed=self.args.seed,
                trainer=self
            )
        else:
            return RepeatSampler(
                data_source=self.train_dataset,
                mini_repeat_count=self.num_generations,
                batch_size=effective_batch_size // self.num_generations,
                repeat_count=self.num_iterations,
                seed=self.args.seed,
            )

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "train" if self.model.training else "eval"

        # if self.accelerator.process_index == 0:
        #     torch.save(inputs, f"/userhome/Research_HUB/GPG/open-r1/wandb/debug/gpg_trainer_line266_{mode}_inputs.pt")

        if mode == "train":
            if not hasattr(self, '_epoch_iterator'):
                frame = inspect.currentframe().f_back.f_back
                print("frame keys ", frame.f_locals.keys())
                # 查找 epoch_iterator
                epoch_iterator = frame.f_locals.get('epoch_iterator')
                assert epoch_iterator is not None
                self._epoch_iterator = epoch_iterator
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

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
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        # old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        # coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        # coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        # per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        # per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        # per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # 在报错前加入调试打印
        # print(f"[RANK {self.accelerator.process_index}] per_token_logps shape: {per_token_logps.shape}")
        # print(f"[RANK {self.accelerator.process_index}] advantages shape: {advantages.shape}")
        per_token_loss = - per_token_logps * advantages.unsqueeze(1)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        #Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        #
        # is_clipped = (per_token_loss1 < per_token_loss2).float()
        # clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        # self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        if self.args.adjust_gd:
            loss = loss / self._metrics[mode]['inverse_alpha'][-1]

        return loss

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
            # if self.accelerator.process_index==0:
            #     torch.save(inputs,f"/userhome/Research_HUB/GPG/open-r1/wandb/debug/gpg_trainer_line747_{mode}_inputs.pt")
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
            # print(f"[DEBUG][RANK {self.accelerator.process_index}] lin922 {mode} rewards.shape:{rewards.shape}")
            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            # print(f"[DEBUG][RANK {self.accelerator.process_index}] lin925 {mode} mean_grouped_rewards.shape:{mean_grouped_rewards.shape},  self.num_generations={self.num_generations}")
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
            # print(f"[DEBUG][RANK {self.accelerator.process_index}] line951 {mode} process_slice:{process_slice}")
            my_rewards = rewards[process_slice] #16

            # if self.accelerator.process_index==0:
            #     print(f"[DEBUG] line-954 {mode} my_rewards:{my_rewards.shape if my_rewards else None}, rewards:{rewards.shape if rewards else None}")

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

            # if self.accelerator.process_index == 0: # [DEBUG] line-990 eval _b_valid:8, _b_ident:8,identical_rewards:torch.Size([8])
            #     print(f"[DEBUG] line-990 {mode} _b_valid:{_b_valid}, _b_ident:{_b_ident},identical_rewards:{identical_rewards.shape if identical_rewards else None}")

            new_rewards = merge(valid_rewards, new_rewards)
            # print(
            #     f"[DEBUG][RANK {self.accelerator.process_index}] lin996 {mode} rewards.shape:{rewards.shape}")
            new_prompt_mask = merge_with_padding(valid_prompt_mask, new_prompt_mask, 0, left_pad=True)
            new_prompt_ids = merge_with_padding(valid_prompt_ids, new_prompt_ids, self.processing_class.pad_token_id, left_pad=True)
            new_completion_mask = merge_with_padding(valid_completion_mask, new_completion_mask, 0, left_pad=False)
            new_completion_ids = merge_with_padding(valid_completion_ids, new_completion_ids, self.processing_class.pad_token_id, left_pad=False)

            # if self.accelerator.process_index == 0 and new_rewards and valid_rewards:#[DEBUG] line-1001 eval new_rewards:torch.Size([8]), valid_rewards:torch.Size([8])
            #     print(f"[DEBUG] line-1001 {mode} new_rewards:{new_rewards.shape}, valid_rewards:{valid_rewards.shape}")

            if n_valid_samples < self.args.min_inverse_alpha * num_samples and mode == "train": # 加了  and mode == "train"
                logger.info(f"keep generating more examples: the {n_gen}-th mini-batch")
                n_gen += 1

            else:
                # 重新组装样本batch
                merge_rewards = merge(identical_rewards, new_rewards)
                rewards =merge_rewards[:len(prompts)]
                # if self.accelerator.process_index == 0:
                #     print(f"[DEBUG] line-1012 {mode} merge_rewards:{merge_rewards.shape if merge_rewards else None}, identical_rewards:{identical_rewards.shape if identical_rewards else None}, new_rewards:{new_rewards.shape if new_rewards else None}, rewards:{rewards.shape if rewards else None}")
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

        # if self.accelerator.process_index == 0:#[DEBUG] line-1033 eval advantages.shape:torch.Size([16]), rewards.shape:torch.Size([16]), mean_grouped_rewards.shape:torch.Size([16])
        #     print(f"[DEBUG] line-1033 {mode} advantages.shape:{advantages.shape if advantages else None}, rewards.shape:{rewards.shape if rewards else None}, mean_grouped_rewards.shape:{mean_grouped_rewards if mean_grouped_rewards else None}")

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

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})


        # if self.accelerator.process_index == 0:
        #     batch = {
        #         "prompt_ids": prompt_ids,
        #         "prompt_mask": prompt_mask,
        #         "completion_ids": completion_ids,
        #         "completion_mask": completion_mask,
        #         "old_per_token_logps": old_per_token_logps,
        #         "ref_per_token_logps": ref_per_token_logps,
        #         "advantages": advantages,
        #     }
        #     torch.save(batch, f"/userhome/Research_HUB/GPG/open-r1/wandb/debug/gpg_trainer_line739_{mode}_batch.pt")

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics
        # if self._metrics[mode].get('num_identical_reward_groups') is not None:
        #     metrics['num_same_reward_groups'] = self._metrics[mode]['num_identical_reward_groups']

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            Trainer.log(self, logs, start_time)
            # super().log(logs, start_time)
        else:  # transformers<=4.46
            Trainer.log(self, logs)
            # super().log(logs)
        self._metrics[mode].clear()

    def _save_checkpoint(self, model, trial):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        import json
        super()._save_checkpoint(model, trial)
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        file_path = os.path.join(output_dir, 'data_weight.json')
        if self.accelerator.is_main_process:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(dict(self.data_weight), file, ensure_ascii=False, indent=4)
            print(f"data_weight 已成功保存到 {file_path}")


    def get_local_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None
        end = False
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                end = True
                break

        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()

        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.item()

        return batch_samples, num_items_in_batch, end


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
