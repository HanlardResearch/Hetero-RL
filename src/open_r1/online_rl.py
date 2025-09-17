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
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from sympy.abc import alpha
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint, speed_metrics

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from open_r1.utils.data_utils import custom_loading_dataset
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from torch.utils.data import DataLoader, Dataset
import time
import math
from transformers.debug_utils import DebugOption
from transformers.utils import is_torch_xla_available
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
from typing import Dict, List, Any
from open_r1.rewards import accuracy_reward_lv35
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.trainer.grpo_trainer import RepeatSampler
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import re
from contextlib import nullcontext
if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
from transformers import (
    Trainer,
    is_wandb_available,
)
if is_wandb_available():
    import wandb
logger = logging.getLogger(__name__)

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

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])

class OnlineRLTrainer(GRPOTrainer):
    """
    online RL trainer for GRPO/GSPO/EqQ
    """
    def __init__(self, *args, **kwargs):
        self.metric_key_prefix = ""
        self.batch_ids = 0
        super().__init__(*args, **kwargs)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # # Compute the KL divergence between the model and the reference model
        # if self.beta != 0.0:
        #     ref_per_token_logps = inputs["ref_per_token_logps"]
        #     per_token_kl = (
        #         torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        #     )

        # Compute the importance weights
        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )

        ################## 样本-level 的P和Q ##################
        sampler_seq_lopp =  (old_per_token_logps * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)
        learner_seq_lopp = (per_token_logps * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)

        ## policy entropy https://arxiv.org/pdf/2505.22617
        sampler_entropy = -sampler_seq_lopp.detach().mean()
        learner_entropy = -learner_seq_lopp.detach().mean()

        avg_sampler_seq_p = sampler_seq_lopp.exp().mean().detach()
        std_sampler_seq_p = sampler_seq_lopp.exp().std().detach()
        adv_std = advantages.std()
        learner_seq_p = learner_seq_lopp.exp()
        sampler_seq_p = sampler_seq_lopp.exp()
        normlized_q = sampler_seq_p.detach() / (sampler_seq_p.sum().detach())
        E_qP =  (normlized_q * learner_seq_p).sum()
        E_qQ =  (normlized_q * sampler_seq_p).sum()

        # Compute the loss
        if self.loss_type in ["grpo", "bnpo", "dr_grpo","delta_ln"]:
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

            # if self.beta != 0.0:
            #     per_token_loss = per_token_loss + self.beta * per_token_kl

            if self.loss_type == "grpo":
                loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            elif self.loss_type == "bnpo":
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            elif self.loss_type == "dr_grpo":
                loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            elif self.loss_type == "delta_ln": #∆L Normalization https://arxiv.org/pdf/2509.07558
                alpha = 0.75  # hyper-params (In paper: α = 0.75 for Math, α = 1 for CountDown)
                ## $L_i^{-\alpha}$
                L_alpha  = completion_mask.sum(-1).clamp(min=1)**(-alpha)
                ## $\Sigma_{i=1}^{i=Batch-size}$ $L_i^{-\alpha}$ * M
                LM = L_alpha.sum() *  self.max_completion_length
                coef_deltaL = (L_alpha/LM).unsqueeze(1) # [bs, sql]
                loss = (coef_deltaL * per_token_loss * completion_mask).sum()


        elif self.loss_type in ["EqP", "gepo", "gspo"]:
            if self.loss_type == "EqP":
                coef_1 = learner_seq_p / E_qP
            elif self.loss_type == "gepo":
                coef_1 = learner_seq_p / E_qQ
            elif self.loss_type == "gspo":
                coef_1 = learner_seq_p / sampler_seq_p
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_seq_loss1 = coef_1 * advantages
            per_seq_loss2 = coef_2 * advantages
            per_seq_loss = -torch.min(per_seq_loss1, per_seq_loss2)
            loss = per_seq_loss.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


        ################## 重要性权重 ##################
        ratio_grpo = torch.exp(per_token_logps.detach() - old_per_token_logps)
        ratio_gspo = learner_seq_p.detach()/sampler_seq_p.detach()
        ratio_pEqQ = learner_seq_p.detach()/E_qQ.detach()
        ratio_pEqP = learner_seq_p.detach()/E_qP.detach()

        ############################ 各个 ratio 的方差 #########################
        # per_token_q = torch.exp(old_per_token_logps)
        # mean_token_q = (per_token_q * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)
        # normlized_token_q = per_token_q.detach() / (mean_token_q.sum().detach())
        # ratio_grpo = torch.exp(per_token_logps.detach() - old_per_token_logps)
        # var_ratio_grpo = (ratio_grpo.square() * normlized_token_q * completion_mask).sum() - (ratio_grpo * normlized_q * completion_mask).sum().square()
        var_ratio_grpo = ((ratio_grpo * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).var()
        # var_is_ratios_mean = var_is_ratios.nanmean()
        # var_is_ratios_std = var_is_ratios.std()
        var_ratio_gspo = (ratio_gspo.square() * normlized_q).sum() - (ratio_gspo * normlized_q).sum().square()
        var_P_EqQ =  (ratio_pEqQ.square() * normlized_q).sum() - (ratio_pEqQ * normlized_q).sum().square()
        var_P_EqP =  (ratio_pEqP.square() * normlized_q).sum() - (ratio_pEqP * normlized_q).sum().square()
        if self.loss_type in ["grpo", "bnpo", "dr_grpo","delta_ln"]:
            # var_coef1 = (coef_1.detach().square() * normlized_token_q  * completion_mask).sum() - (coef_1.detach() * normlized_token_q * completion_mask).sum().square()
            # var_coef2 = (coef_2.detach().square() * normlized_token_q  * completion_mask).sum() - (coef_2.detach() * normlized_token_q * completion_mask).sum().square()
      
            var_coef1 = ((coef_1 * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).var()
            var_coef2 = ((coef_2 * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).var()
        else:
            var_coef1 = (coef_1.detach().square() * normlized_q).sum() - (coef_1.detach() * normlized_q).sum().square()
            var_coef2 = (coef_2.detach().square() * normlized_q).sum() - (coef_2.detach() * normlized_q).sum().square()

        
        # Log the metrics
        mode = "train" if self.model.training else "eval"

        ########################## WANDB 显示的统计量 #######################
        self._metrics[mode]["ratio/mean"].append(coef_1.nanmean().item())
        self._metrics[mode]["ratio/max"].append(nanmax(coef_1).item())
        self._metrics[mode]["ratio/min"].append(nanmin(coef_1).item())
        # self._metrics[mode]["var_ratio_grpo"].append(var_ratio_grpo.item())
        # self._metrics[mode]["var_ratio_pq"].append(var_ratio_gspo.item())
        # self._metrics[mode]["var_P_EqQ"].append(var_P_EqQ.item())
        # self._metrics[mode]["var_P_EqP"].append(var_P_EqP.item())

        # self._metrics[mode]["sts_var/ratio_grpo"].append(ratio_grpo.var().item())
        self._metrics[mode]["sts_var/ratio_pq"].append(ratio_gspo.var().item())
        # self._metrics[mode]["sts_var/ratio_pEqQ"].append(ratio_pEqQ.var().item())
        # self._metrics[mode]["sts_var/ratio_pEqP"].append(ratio_pEqP.var().item())

        ## policy entropy https://arxiv.org/pdf/2505.22617
        self._metrics[mode]["sampler_entropy"].append(sampler_entropy.item())
        self._metrics[mode]["learner_entropy"].append(learner_entropy.item())

        self._metrics[mode]["var_coef1"].append(var_coef1.item())
        self._metrics[mode]["var_coef2"].append(var_coef2.item())
        # self._metrics[mode]["ratio_grpo"].append(ratio_grpo.nanmean().item())
        self._metrics[mode]["ratio_pq"].append(ratio_gspo.nanmean().item())
        # self._metrics[mode]["ratio_pEqQ"].append(ratio_pEqQ.nanmean().item())
        # self._metrics[mode]["ratio_pEqP"].append(ratio_pEqP.nanmean().item())
        self._metrics[mode]["adv_std"].append(adv_std.item())
        self._metrics[mode]["avg_sampler_seq_p"].append(avg_sampler_seq_p.item())
        self._metrics[mode]["std_sampler_seq_p"].append(std_sampler_seq_p.item())

        # if self.beta != 0.0:
        #     mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        #     self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        if self.loss_type in ["grpo", "bnpo", "dr_grpo","delta_ln"]:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
            high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
            clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                # print(f"[debug] line329",eval_dataset.items())
                # print(f"[debug] line330 {eval_dataset_name}")
                # print(f"[debug] line331",_eval_dataset)

                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        self.metric_key_prefix = metric_key_prefix
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        
        self.log(output.metrics)
        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)
        
        return output.metrics

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        ground_truth = [x["answer"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
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
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
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
                            generation_kwargs=self.args.generation_kwargs,
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

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
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
            batch_var = mean_grouped_rewards.var()
            a = (batch_mean*(1-batch_mean)/batch_var-1)*batch_mean if batch_var > 0 else torch.tensor(0.0, device=device)
            b = (batch_mean*(1-batch_mean)/batch_var-1)*(1-batch_mean) if batch_var > 0 else torch.tensor(0.0, device=device)
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
            if mode == "eval":
                rewards_str = self.metric_key_prefix + "_rewards" if self.metric_key_prefix != "eval" else "rewards"
            else:
                rewards_str = "rewards"

            self._metrics[mode][f"{rewards_str}/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"{rewards_str}/{reward_func_name}/std"].append(std_rewards)
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
            ground_truth_to_log = gather_object(ground_truth)
            if self.accelerator.is_main_process:
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.batch_ids)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "ground_truth": ground_truth_to_log,
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
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
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
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir):
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    # tokenizer.padding_side  = 'left' # fix the last evalute() issue? not tested yet.
    # handle dataset
    # Load the dataset
    if 'simplelr_qwen_level3to5' in script_args.dataset_name:
        dataset = custom_loading_dataset(script_args.dataset_name, max_length=training_args.max_prompt_length,
                                         tokenizer=tokenizer)

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
        # prompt.append({"role": "user", "content": example["problem"]+"/no_think"})
        return {"prompt": prompt}
    
    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    
    if training_args.eval_strategy == "no":
        eval_dataset = None
    else:
        eval_dataset = dataset[script_args.dataset_test_split]
        if script_args.use_benchmark:
            amc23_data = load_dataset("/extrahome0/HF_datasets/amc23")
            amc23_data = amc23_data.rename_column("question", "problem")[script_args.dataset_test_split].map(make_conversation)
            # amc23_data = amc23_data.rename_column("answer", "solution")[script_args.dataset_test_split].map(make_conversation)
            aime_2024_data = load_dataset("/extrahome0/HF_datasets/aime_2024")[script_args.dataset_train_split].map(make_conversation)
            aime_2025_data = load_dataset("/extrahome0/HF_datasets/aime_2025")[script_args.dataset_train_split].map(make_conversation)
            math_500_data = load_dataset("/extrahome0/HF_datasets/MATH-500")[script_args.dataset_test_split].map(make_conversation)

            eval_data = {
                "validaion": eval_dataset,
                "amc23": amc23_data,
                "aime_2024": aime_2024_data,
                "aime_2025": aime_2025_data,
                "math_500": math_500_data
            }
            eval_dataset = eval_data
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
    trainer = OnlineRLTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    resume_from_checkpoint = None
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint == "True":
            resume_from_checkpoint = True
        elif os.path.exists(training_args.resume_from_checkpoint):
            resume_from_checkpoint = training_args.resume_from_checkpoint
            logger.info(f"Checkpoint detected, resuming training at {resume_from_checkpoint=}.")
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
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
    
    print(f"finished!")


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
