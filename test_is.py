import re
from trl.trainer.utils import pad
from tqdm import tqdm
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import matplotlib.pyplot as plt
import torch.nn.functional as F
from contextlib import nullcontext
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from vllm import LLM, SamplingParams
from trl.trainer.utils import selective_log_softmax
from trl.extras.profiling import profiling_context
import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

def custom_loading_dataset(dataset_name, train_name='train.parquet', test_name='test.parquet', max_length=512, tokenizer=None):
    """
    Load and preprocess a dataset from Parquet files, and filter out samples exceeding a specified length.

    Args:
        dataset_name (str): The base directory of the dataset.
        train_name (str, optional): The name of the training file. Defaults to 'train.parquet'.
        test_name (str, optional): The name of the test file. Defaults to 'test.parquet'.
        max_length (int, optional): Maximum length of the samples to keep. Defaults to 512.
        tokenizer (str, optional): tokenizer to use. Defaults to 'bert-base-uncased'.

    Returns:
        DatasetDict: A dictionary-like object containing the training and test datasets.
    """
    # 定义数据文件路径
    train_path = os.path.join(dataset_name, train_name)
    test_path = os.path.join(dataset_name, test_name)


    # 定义一个函数来计算文本的长度
    def get_length(text):
        inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
        return inputs["input_ids"].shape[1]

    # 读取训练数据
    try:
        train_data = pd.read_parquet(train_path)
        train_data['split'] = 'train'  # 添加 split 列
    except FileNotFoundError:
        raise FileNotFoundError(f"Training file not found at {train_path}")

    # 读取测试数据
    try:
        test_data = pd.read_parquet(test_path)
        test_data['split'] = 'test'  # 添加 split 列
    except FileNotFoundError:
        print(f"Test file not found at {test_path}. Skipping test data.")
        test_data = None

    # 定义列名映射
    column_mapping = {
        'ground_truth_answer': 'ground_truth',
        'subject': 'topic',
        'target': 'solution',
        # 'data_source': 'source',
        'input': 'instruction',
        # 'ability': 'skill',
        # 'reward_model': 'reward',
        # 'extra_info': 'metadata',
        'question': 'problem'
    }


    # 重命名列
    train_data.rename(columns=column_mapping, inplace=True)

    if test_data is not None:
        test_data.rename(columns=column_mapping, inplace=True)


    # 计算每个样本的长度
    train_data['length'] = train_data['instruction'].apply(get_length)
    if test_data is not None:
        test_data['length'] = test_data['instruction'].apply(get_length)

    # 过滤掉超过 max_length 的样本
    train_data = train_data[train_data['length'] <= max_length]
    if test_data is not None:
        test_data = test_data[test_data['length'] <= max_length]

    # 转换为 Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_data)
    if test_data is not None:
        test_dataset = Dataset.from_pandas(test_data)
    else:
        test_dataset = None

    # 创建 DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    return dataset_dict
def make_conversation(example):
    prompt = []
    system_prompt = "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags."
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": example["problem"]})


    # prompt = example["problem"] + " The reasoning process MUST BE enclosed within <think> and </think> tags. Please reason step by step, and put your final answer within \\boxed{}."
    # if add_think:
    #     prompt += " /think"

    return {"prompt": prompt}


def pre_process(completions):
    """retrieve the completion content from input"""
    if  isinstance(completions[0],(list,)):
        completion_contents = [completion[0]["content"] for completion in completions]
    elif isinstance(completions[0],(dict)):
        completion_contents = [completion["content"] for completion in completions]
    else:
        completion_contents = [completion for completion in completions]
    return completion_contents

def accuracy_reward_lv35(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    # if isinstance(completions[0],(dict)):
    #     contents = [completion["content"] for completion in completions]
    # else:
    #     contents = [completion for completion in completions]
    contents = pre_process(completions)
    rewards = []
    for content, sol in zip(contents, solution):
        box_sol = "$\\\\boxed{}$".format(sol)
        try:
            gold_parsed = parse(
                box_sol,
                extraction_mode="first_match",
            )
        except TimeoutError:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[Rank  {rank}] gold parse timeout | content='{content}' | sol='{sol}' | box_sol='{box_sol}'")
            rewards.append(1.0)
            continue
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            try:
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed="all",
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                # print(f'answer_parsed:{answer_parsed}')
                # if len(anxswer_parsed) == 0:
                #     print(f"answer_parsed is None | content='{content}' | sol='{sol}'")
            except TimeoutError:
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(f"[Rank {rank}] answer parse timeout | content='{content}' | sol='{sol}'")
                rewards.append(0.0)
                continue
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("accuracy_reward_lv35: Failed to parse gold solution: ", box_sol)
        rewards.append(reward)
        

    return torch.Tensor(rewards)

def _get_per_token_logps(temperature, model, input_ids, attention_mask, logits_to_keep, batch_size=None) -> torch.Tensor:
    batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
    all_logps = []
    for i in tqdm(range(0, input_ids.size(0), batch_size)):
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
        logits = logits / temperature
        logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
        all_logps.append(logps)
    return torch.cat(all_logps, dim=0)

def move_to_vllm(model, llm):
    """synchronize the weight of HF model to vllm model"""
    for name, param in model.named_parameters():
        with nullcontext([param]):
            llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights([(name, param.data)])
    llm.reset_prefix_cache()
    print('vllm updated!')


def stat(ind_p, ind_q, enable_seq_loss):
    """calculate statistics on top of sampling result"""
    log_p, log_q = log_probs[ind_p],log_probs[ind_q]
    adv_p, adv_q = advantages_list[ind_p],advantages_list[ind_q]
    # adv_p = adv_p.to(device)
    # adv_q = adv_q.to(device)
    if enable_seq_loss:
        sampler_seq_lopp =  (log_q * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)
        learner_seq_lopp = (log_p * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)
        avg_sampler_seq_p = sampler_seq_lopp.exp().mean().detach()
        std_sampler_seq_p = sampler_seq_lopp.exp().std().detach()
        learner_seq_p = learner_seq_lopp.exp()

        # coef_1 = learner_seq_p / (avg_sampler_seq_p*(1-std_sampler_seq_p) + std_sampler_seq_p * learner_seq_p.detach() )
        is_ratios = learner_seq_p / avg_sampler_seq_p
        per_token_kl = torch.exp(log_q - log_p) - (log_q - log_p) - 1
        mean_kl = (per_token_kl * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        kl_mean = mean_kl.double().nanmean()
        print(f'kl_mean: {kl_mean}')
        final_adv_p = adv_p
        final_adv_q = is_ratios * adv_q
        cos_sim = F.cosine_similarity(final_adv_p, final_adv_q, dim=0).mean()
        print(f'cosine_similarity: {cos_sim}')
    else:
        is_ratios = torch.exp(log_p - log_q)
        per_token_kl = torch.exp(log_q - log_p) - (log_q - log_p) - 1
        mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        kl_mean = mean_kl.double().nanmean()
        
        # loss = (is_ratios * rewards).mean()
        final_adv_p = adv_p.unsqueeze(1).expand_as(is_ratios)
        final_adv_q = is_ratios * adv_q.unsqueeze(1)
        cos_sim = F.cosine_similarity(final_adv_p, final_adv_q, dim=1).mean()

        # extend to 64-bit to avoid inf problem for mse calculation
        final_adv_p = final_adv_p.double()
        final_adv_q = final_adv_q.double()
    # diff = adv_p.unsqueeze(1) - is_ratios * adv_q.unsqueeze(1)
    # mse = ((final_adv_p - final_adv_q) ** 2).mean()
    print(f'kl_mean: {kl_mean}')
    print(f'cosine_similarity: {cos_sim}')
    mse = F.mse_loss(final_adv_p, final_adv_q)
    print(f'mse: {mse}')
    corr_global = torch.corrcoef(torch.stack([
        final_adv_p.reshape(-1),
        final_adv_q.reshape(-1)
    ]))[0, 1]
    print(f'global corr between {model_list[ind_p]} and {model_list[ind_q]}: {corr_global}')
    return cos_sim.cpu(), mse.cpu(), corr_global.cpu(), kl_mean.item()

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

temperature=0.6
top_p=0.95
N=64
top_k=20
max_length = 2048
num_generations=8
begin_ind=0
end_ind=begin_ind+N
scale_rewards = False
solutions = []
prompts_text = []
sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_length)

# Initialize the vLLM engine
llm = LLM(model="/extrahome0/HF_models/Qwen/Qwen3-1.7B",gpu_memory_utilization=0.8)
tokenizer =AutoTokenizer.from_pretrained("/extrahome0/HF_models/Qwen/Qwen3-1.7B", trust_remote_code=True)
data_path = "/extrahome0/HF_datasets/open-r1/simplelr_qwen_level3to5"
model_id = "/extrahome0/HF_models/Qwen/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

device = model.device
batch_size = 8
max_prompt_length = 768
seed = 42
mode = "test"

# data pre-process
dataset = custom_loading_dataset(data_path, max_length=max_prompt_length, tokenizer=tokenizer)
dataset = dataset.map(make_conversation)
for split in dataset:
    if "messages" in dataset[split].column_names:
        dataset[split] = dataset[split].remove_columns("messages")

current_dataset = dataset['train'] if mode == "train" else dataset['test']
current_dataset = current_dataset.shuffle(seed=seed)

for ind in range(begin_ind, end_ind):
    for _ in range(num_generations):
        prompts_text.append("<|im_start|>system\nYou are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. Please put your final answer within \\boxed{}. Also, indicate that it is the answer.<|im_start|>user\n" + current_dataset[ind]['problem'] + "<|im_end|>\n<|im_start|>assistant\n")
        solutions.append(current_dataset[ind]['solution'])

prompt_inputs = tokenizer(
    text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
)
prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(device), prompt_inputs["attention_mask"].to(device)
if max_prompt_length is not None:
    # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
    # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
    # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
    prompt_ids = prompt_ids[:, -max_prompt_length :]
    prompt_mask = prompt_mask[:, -max_prompt_length :]
    prompts_text = tokenizer.batch_decode(
        prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    prompts_text = [
        re.sub(rf"^({re.escape(tokenizer.pad_token)})+", "", text) for text in prompts_text
    ]

print(f'number of prompts: {len(prompts_text)}')
model_list = ["gpg_async_weights_global_step_233.pt", "gpg_async_weights_global_step_236.pt", "gpg_async_weights_global_step_248.pt", "gpg_async_weights_global_step_249.pt", "gpg_async_weights_global_step_250.pt", "gpg_async_weights_global_step_251.pt", "gpg_async_weights_global_step_252.pt", "gpg_async_weights_global_step_253.pt", "gpg_async_weights_global_step_254.pt", "gpg_async_weights_global_step_255.pt"]
# def get_logprobs_and_reward(model_list):
log_probs = []
advantages_list = []

# sampling
for model_name in model_list:
    state_dict = torch.load("/userhome/save_dir/AsyncGRPO/4gpus/Async_MoISv6e_1th_cfgv6b/tmp/Qwen3-1.7B/"+model_name, map_location="cpu")[1]
    model.load_state_dict(state_dict)
    move_to_vllm(model, llm)
    all_outputs = llm.generate(prompts_text, sampling_params)
    completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
    completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
    completion_ids = pad(completion_ids, padding_value=tokenizer.pad_token_id)
    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    with torch.no_grad():
        logps = _get_per_token_logps(
            model=model,
            input_ids=prompt_completion_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            temperature=temperature,
            batch_size=batch_size
        )
        log_probs.append(logps)
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    rewards = accuracy_reward_lv35(completions=completions_text, solution=solutions).to(device)
    mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, num_generations).std(dim=1)
    # is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

    # Normalize the rewards to compute the advantages
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
    advantages = rewards - mean_grouped_rewards
    if scale_rewards:
        advantages = advantages / (std_grouped_rewards + 1e-4)
    advantages_list.append(advantages)

cos_sim_list = []
mse_list = []
global_corr_list = []
kl_list = []
ind_p = 9
for ind_q in range(9,1,-1):
    cos_sim, mse, corr_global, kl = stat(ind_p, ind_q, enable_seq_loss=False)
    cos_sim_list.append(cos_sim)
    mse_list.append(mse)
    global_corr_list.append(corr_global)
    kl_list.append(kl)

# 画图
plt.figure(figsize=(8, 5))
plt.subplot(4,1,1)
plt.plot(cos_sim_list,".-")  # 画曲线
plt.legend(["cos_sim"])
plt.grid()
plt.subplot(4,1,2)
plt.plot(mse_list,".-")  # 画曲线
plt.legend(["mse"])
plt.grid()
plt.subplot(4,1,3)
plt.plot(global_corr_list,".-")  # 画曲线
plt.legend(["global_corr"])
plt.grid()
plt.subplot(4,1,4)
plt.plot(kl_list,".-")  # 画曲线
plt.legend(["kl"])
plt.grid()
plt.xlabel(f'delay_delta from {model_list[ind_p]}')
plt.tight_layout()
plt.savefig(f"/code/tmp/status/cosine_similarity_{model_list[ind_p]}_pairs.png")
plt.savefig(f"/code/tmp/status/cosine_similarity_{model_list[ind_p]}_pairs.pdf")