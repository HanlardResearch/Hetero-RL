# coding=utf-8
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

from dataclasses import dataclass, field
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    # vllm_mode: Optional[str] = field(default="colocate", metadata={"help": "Vllm mode to consume."})
    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )

@dataclass
class HeteroRLConfig(GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    scale_batch: bool = field(default=False, metadata={"help": "divide std use group or batch std."})
    adjust_lr: bool = field(default=False, metadata={"help": "adjust lr based on the valid sample within a batch."})
    weighted_sample: bool = field(default=False, metadata={"help": "weight sample based on the running acc."})
    adjust_gd: bool = field(default=False, metadata={"help": "adjust gd  based on the valid sample within a batch."})
    sample_strategy: Optional[str] = field(
        default="uniform",
        metadata={"help": ("hard, uniform.")},
    )
    min_inverse_alpha: float = field(default=0.5, metadata={"help": "minimum inverse alpha"})



@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )


@dataclass
class GRPOScriptArguments(trl.ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'ioi_code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={
            "help": "for each generation, evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases. Useful to avoid overloading the eval server + save time on wrong solutions"
        },
    )
    wandb_name: Optional[str] = field(
        default=None,
        metadata={"help": "name of wandb run"},
    )
    cppo_beta: float = field(
        default=0.0,
        metadata={"help": "beta of CPPO-KL loss"},
    )
    max_diff_step:int = field(
        default=32,
        metadata={"help": "max tolerable delay steps"},
    )
    use_benchmark: bool = field(
        default=False, 
        metadata={"help": "Whether enable the benchmark for evaluation."},
    )
    use_think: bool = field(
        default=False,
        metadata={"help": "Whether enable the think mode."},
    )
    system_prompt_think: str = field(
        default="You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. Please put your final answer within \\boxed{}. Also, indicate that it is the answer.",
    )
    system_prompt_nothink: str = field(
        default="You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. Please put your final answer within \\boxed{}. Also, indicate that it is the answer.",
    )
@dataclass
class HeteroRLScriptArguments(GRPOScriptArguments):
    """
    Script arguments for the HeteroRL training script.
    """
    delay_sampler: str = field(
        default="lognormal",
        metadata={
            "help": "时间延迟分布 P(t)",
            "choices": ["lognormal", "weibull"],
        },
    )
    lower_bound: float = field(
        default=60.0,
        metadata={"help": "时间延迟下界"},
    )
    upper_bound: float = field(
        default=1920,
        metadata={"help": "时间延迟上界"},
    )
    confidence: float = field(
        default=0.995,
        metadata={"help": "时间延迟采样器的置信度"},
    )
    default_delay: float = field(
        default=60.0,
        metadata={"help": "时间延迟-基础时间延迟 服从于时间延迟分布 P(t)"},
    )
    sampler_id: int = field(
        default=0,
        metadata={"help": "The id of current sampler."},
    )
    num_samplers: int = field(
        default=1,
        metadata={"help": "The amount of samplers."},
    )
    online_mode: bool = field(
        default=False, 
        metadata={"help": "Whether force the sampler to use the latest weight for sampling."},
    )
    wandb_name: Optional[str] = field(
        default=None,
        metadata={"help": "说明当前wandb-run的功能"},
    )
