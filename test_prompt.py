# from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import os
import pandas as pd
import torch
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_ratio_of_ones(lst):
    total_count = len(lst)  # 列表中所有元素的数量
    ones_count = lst.count(1)  # 值为1的元素数量
    if total_count == 0:
        return 0  # 避免除以0的情况，如果列表为空，则返回0
    ratio = ones_count / total_count  # 计算比率
    return ratio

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
#                 print(f'answer_parsed:{answer_parsed}')
                if len(answer_parsed) == 0:
                    reward = -1.0
                else:
                    reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("accuracy_reward_lv35: Failed to parse gold solution: ", box_sol)
        rewards.append(reward)
        

    return rewards


tokenizer =AutoTokenizer.from_pretrained("/extrahome0/HF_models/Qwen/Qwen3-1.7B")
data_path = "/extrahome0/HF_datasets/open-r1/simplelr_qwen_level3to5"
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
dataset = custom_loading_dataset(data_path, max_length=768, tokenizer=tokenizer)
dataset = dataset.map(make_conversation)
for split in dataset:
    if "messages" in dataset[split].column_names:
        dataset[split] = dataset[split].remove_columns("messages")
dataset['train'] = dataset['train'].shuffle(seed=42)
# dataset['train']

# Initialize the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("/extrahome0/HF_models/Qwen/Qwen3-1.7B")

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

# Initialize the vLLM engine
llm = LLM(model="/extrahome0/HF_models/Qwen/Qwen3-1.7B")

sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048)
# problem = "In terms of $\\pi$, what is the area of the circle defined by the equation $2x^2+2y^2+10x-6y-18=0$?"
num = 100
reward_from_system1 = []
reward_from_system2 = []
len_of_completion_from_system1 = []
len_of_completion_from_system2 = []
for ind in tqdm(range(num)):
    problem = dataset['train'][ind]['problem']
    solution = dataset['train'][ind]['solution']
    text_with_system1 = "<|im_start|>system\nYou are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. Please put your final answer within \\boxed{}. Also, indicate that it is the answer.<|im_start|>user\n" + problem + "<|im_end|>\n<|im_start|>assistant\n"
    text_with_system2 = "<|im_start|>system\nYou are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST provide the answer within \\boxed{} on its own line after \"Answer:\" before your reasoning process and then supplement the reasoning process as an internal monologue.<|im_start|>user\n" + problem + "/no_think<|im_end|>\n<|im_start|>assistant\n"
    # text_with_system2 = "<|im_start|>system\nYou are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST provide the answer within \\boxed{} on its own line after \"Answer:\" before your reasoning process and then supplement the reasoning process as an internal monologue. 请把答案紧接着放在<think>后面，因为我想尽快看到答案<|im_start|>user\n" + problem + "<|im_end|>\n<|im_start|>assistant\n"
    # template_with_system2 = """<|im_start|>system\nYou are a helpful AI assistant. When responding to queries, please follow this format:

    # 1. Start with "Answer:" on a new line.
    # 2. On the next line, provide the final answer enclosed in \\boxed{}.
    # 3. After the boxed answer, begin your detailed reasoning process as an internal monologue, explaining how you arrived at the answer.

    # Ensure the answer is presented clearly and immediately before any explanation.<|im_start|>user\n"""
    # text_with_system2 = template_with_system2 + problem + "<|im_end|>\n<|im_start|>assistant\n"
    # Generate outputs

    output_from_system = llm.generate([text_with_system1,text_with_system2], sampling_params, use_tqdm=False)
#     output_from_system2 = llm.generate([], sampling_params)

    # # Print the outputs.
    # for output_from_system2, output_from_system2 in zip(outputs_from_system1, outputs_from_system2):
#     print("*"*100+"system1"+"*"*100)
    prompt_from_system1 = output_from_system[0].prompt
    completion_from_system1 = output_from_system[0].outputs[0].text
    len_of_completion_from_system1.append(len(completion_from_system1))
#     print(f'length of completion_from_system1:{len(completion_from_system1)}')
    # print(f"Prompt: {prompt_from_system1!r}")
    # print("-"*100)
    # print(f"Generated text: {completion_from_system1!r}")
    # print("*"*100+"system2"+"*"*100)
    prompt_from_system2 = output_from_system[1].prompt
    completion_from_system2 = output_from_system[1].outputs[0].text
    len_of_completion_from_system2.append(len(completion_from_system2))
#     print(f'length of completion_from_system2:{len(completion_from_system2)}')
    # print(f"Prompt: {prompt_from_system2!r}")
    # print("-"*100)
    # print(f"Generated text: {completion_from_system2!r}")
    # print("*"*100+"without_system"+"*"*100)
    rewards = accuracy_reward_lv35(completions=[completion_from_system1,completion_from_system2], solution=[solution,solution])
    reward_from_system1.append(rewards[0])
    reward_from_system2.append(rewards[1])

print(f'think准确率: {calculate_ratio_of_ones(reward_from_system1)}')
print(f'no_think准确率: {calculate_ratio_of_ones(reward_from_system2)}')
mean_len_of_completion_from_system1 = sum(len_of_completion_from_system1)/len(len_of_completion_from_system1)
mean_len_of_completion_from_system2 = sum(len_of_completion_from_system2)/len(len_of_completion_from_system2)
#     print(f'completion_from_system2: {accuracy_reward_lv35(completions=[], solution=[])}')


# 绘制第一组图表
plt.figure(figsize=(24, 6))

# 第一组数据
plt.subplot(2, 1, 1)  # 创建一个1行2列的子图，当前是第一个子图
plt.plot(reward_from_system1, ".-", color="blue", label="think")
plt.plot(reward_from_system2, ".-", color="green", label="no_think")
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
plt.grid(True)
plt.title(f'think accuracy rate: {calculate_ratio_of_ones(reward_from_system1):.2} |  no_think accuracy rate: {calculate_ratio_of_ones(reward_from_system2):.2}')
plt.xlabel("query")
plt.ylabel("reward")

# 绘制第二组图表
plt.subplot(2, 1, 2)  # 创建一个1行2列的子图，当前是第二个子图
plt.plot(len_of_completion_from_system1, ".-", color="red", label="think")
plt.plot(len_of_completion_from_system2, ".-", color="purple", label="no_think")
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
plt.grid(True)
plt.title(f'think mean length: {round(mean_len_of_completion_from_system1)} |  no_think mean length: {round(mean_len_of_completion_from_system2)}')
plt.xlabel("query")
plt.ylabel("length_completion")

# 显示图表
plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域
# plt.show()
plt.savefig(f"/code/tmp/status/prompt_compare.png")
plt.savefig(f"/code/tmp/status/prompt_compare.pdf")
plt.close()
