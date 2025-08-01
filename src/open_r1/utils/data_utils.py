import os
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict


# def custom_loading_dataset(dataset_name, train_name='train.parquet', test_name='test.parquet'):
#     """
#     Load and preprocess a dataset from Parquet files.
#
#     Args:
#         dataset_name (str): The base directory of the dataset.
#         train_name (str, optional): The name of the training file. Defaults to 'train.parquet'.
#         test_name (str, optional): The name of the test file. Defaults to 'test.parquet'.
#
#     Returns:
#         DatasetDict: A dictionary-like object containing the training and test datasets.
#     """
#     # 定义数据文件路径
#     train_path = os.path.join(dataset_name, train_name)
#     test_path = os.path.join(dataset_name, test_name)
#
#     # 读取训练数据
#     try:
#         train_data = pd.read_parquet(train_path)
#         train_data['split'] = 'train'  # 添加 split 列
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Training file not found at {train_path}")
#
#     # 读取测试数据
#     try:
#         test_data = pd.read_parquet(test_path)
#         test_data['split'] = 'test'  # 添加 split 列
#     except FileNotFoundError:
#         print(f"Test file not found at {test_path}. Skipping test data.")
#         test_data = None
#
#     # 定义列名映射
#     column_mapping = {
#         'gt_answer': 'ground_truth',
#         'subject': 'topic',
#         'target': 'solution',
#         'data_source': 'source',
#         'prompt': 'instruction',
#         'ability': 'skill',
#         'reward_model': 'reward',
#         'extra_info': 'metadata',
#         'question': 'problem'
#     }
#
#     # 重命名列
#     train_data.rename(columns=column_mapping, inplace=True)
#     if test_data is not None:
#         test_data.rename(columns=column_mapping, inplace=True)
#
#     # 转换为 Hugging Face Dataset
#     train_dataset = Dataset.from_pandas(train_data)
#     if test_data is not None:
#         test_dataset = Dataset.from_pandas(test_data)
#     else:
#         test_dataset = None
#
#     # 创建 DatasetDict
#     dataset_dict = DatasetDict({
#         'train': train_dataset,
#         'test': test_dataset
#     })
#
#     return dataset_dict


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