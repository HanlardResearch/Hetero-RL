from transformers import AutoTokenizer
import torch
from open_r1.utils.data_utils import custom_loading_dataset

# 加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-Math-7B")
dataset = custom_loading_dataset("datas/SimpleRL-Zoo-Data/simplelr_qwen_level3to5", tokenizer=tokenizer)


def make_conversation_math35(example):
    prompt = []
    # prompt.append({"role": "user", "content": example["instruction"][0]['content']})
    prompt = example["instruction"][0]['content']
    # prompt.append({"role": "user", "content": example["problem"]})
    return {"prompt": prompt}

dataset = dataset.map(make_conversation_math35)

# 初始化最大长度变量
max_length = 0

# 遍历数据集，计算每个样本的长度
for text in dataset['train']:
    # 使用分词器对文本进行编码
    text = text['prompt']
    print(text)
    inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
    # 获取输入的长度
    length = inputs["input_ids"].shape[1]
    # 更新最大长度
    if length > max_length:
        max_length = length

print(f"Maximum length after tokenization: {max_length}")