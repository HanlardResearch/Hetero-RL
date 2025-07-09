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

from typing import Dict, Any, Union

logger = logging.getLogger(__name__)


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
            # logger.info(f"inverse_alpha: {self._metrics['train']['inverse_alpha'][-1]}")
            # 返回准备好的、在正确设备上的数据
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
        eval_dataset=eval_dataset,
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


