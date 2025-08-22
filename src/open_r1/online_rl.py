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
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from open_r1.utils.data_utils import custom_loading_dataset
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

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
        avg_sampler_seq_p = sampler_seq_lopp.exp().mean().detach()
        std_sampler_seq_p = sampler_seq_lopp.exp().std().detach()
        adv_std = advantages.std()
        learner_seq_p = learner_seq_lopp.exp()
        sampler_seq_p = sampler_seq_lopp.exp()
        normlized_q = sampler_seq_p.detach() / (sampler_seq_p.sum().detach())
        E_qP =  (normlized_q * learner_seq_p).sum()
        E_qQ =  (normlized_q * sampler_seq_p).sum()

        # Compute the loss
        if self.loss_type in ["grpo", "bnpo", "dr_grpo"]:
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
        elif self.loss_type in ["EqP", "EqQ", "gspo"]:
            if self.loss_type == "EqP":
                coef_1 = learner_seq_p / E_qP
            elif self.loss_type == "EqQ":
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
        if self.loss_type in ["grpo", "bnpo", "dr_grpo"]:
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
        self._metrics[mode]["var_ratio_grpo"].append(var_ratio_grpo.item())
        self._metrics[mode]["var_ratio_pq"].append(var_ratio_gspo.item())
        self._metrics[mode]["var_P_EqQ"].append(var_P_EqQ.item())
        self._metrics[mode]["var_P_EqP"].append(var_P_EqP.item())

        self._metrics[mode]["sts_var/ratio_grpo"].append(ratio_grpo.var().item())
        self._metrics[mode]["sts_var/ratio_pq"].append(ratio_gspo.var().item())
        self._metrics[mode]["sts_var/ratio_pEqQ"].append(ratio_pEqQ.var().item())
        self._metrics[mode]["sts_var/ratio_pEqP"].append(ratio_pEqP.var().item())

        self._metrics[mode]["var_coef1"].append(var_coef1.item())
        self._metrics[mode]["var_coef2"].append(var_coef2.item())
        self._metrics[mode]["ratio_grpo"].append(ratio_grpo.nanmean().item())
        self._metrics[mode]["ratio_pq"].append(ratio_gspo.nanmean().item())
        self._metrics[mode]["ratio_pEqQ"].append(ratio_pEqQ.nanmean().item())
        self._metrics[mode]["ratio_pEqP"].append(ratio_pEqP.nanmean().item())
        self._metrics[mode]["adv_std"].append(adv_std.item())
        self._metrics[mode]["avg_sampler_seq_p"].append(avg_sampler_seq_p.item())
        self._metrics[mode]["std_sampler_seq_p"].append(std_sampler_seq_p.item())

        # if self.beta != 0.0:
        #     mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        #     self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        if self.loss_type in ["grpo", "bnpo", "dr_grpo"]:
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
        return {"prompt": prompt}

    def make_conversation_math35(example):
        prompt = []
        prompt.append({"role": "user", "content": example["instruction"][0]['content']})
        # prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    # if 'simplelr_qwen_level3to5' in script_args.dataset_name:
    #     dataset = dataset.map(make_conversation_math35)
    # else:
    #     dataset = dataset.map(make_conversation)
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
    trainer = OnlineRLTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
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
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
