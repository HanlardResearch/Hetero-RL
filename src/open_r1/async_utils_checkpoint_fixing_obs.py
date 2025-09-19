import logging
import os
import pickle
from collections import defaultdict
from email.policy import default
from pathlib import Path
from accelerate.utils import broadcast, wait_for_everyone

import torch
import torch.nn as nn
from transformers import TrainerCallback, TrainingArguments
from typing import Optional, Dict, Any
import os
from trl.extras.profiling import profiling_decorator
import time
import uuid
from pathlib import Path
from typing import Optional
import random
import obs

SCAN_SLEEP_TIME = 0.1 #every 0.5 seconds scan
SCAN_DIRECTORY = 'obs://yzx/'
bucket_name = SCAN_DIRECTORY[6:].split('/',1)[0]

hf_obs_dir_format = lambda x: x if x.endswith('/') else (x+'/')

def obs_path_to_key(obs_path, bucket_name=bucket_name):
    tmp0 = 'obs://'+bucket_name
    assert obs_path.startswith(tmp0)
    ret = obs_path[(len(tmp0)+1):]
    return ret

def parse_obs_path(obs_path):
    assert obs_path.startswith('obs://')
    obs_path = obs_path[6:]
    bucket_name,key = obs_path.split('/',1)
    return bucket_name, key

def obs_isfile(obs_client, obs_path):
    bucket,key = parse_obs_path(obs_path)
    assert not key.endswith('/')
    ret = obs_client.getObjectMetadata(bucket, key)['status']==200
    return ret

def obs_upload_file(obs_client, local_path, obs_path):
    obs_path = obs_path.rstrip('/')
    bucket_name,key = parse_obs_path(obs_path)
    response = obs_client.uploadFile(bucket_name, key, local_path)
    if response.status!=200:
        print(f'[{response.status}] fail for uploading "{local_path}"')
    return response.status

def obs_download_file(obs_client, obs_path, local_path):
    obs_path = obs_path.rstrip('/')
    bucket_name,key = parse_obs_path(obs_path)
    if os.path.isdir(local_path):
        local_path = os.path.join(local_path, obs_path.rsplit('/')[-1])
    response = obs_client.downloadFile(bucket_name, key, local_path)
    if response.status!=200:
        print(f'[{response.status}] fail for downloading "{obs_path}"')

def obs_listdir(obs_client, obs_path):
    bucket_name,key = parse_obs_path(obs_path)
    key = hf_obs_dir_format(key)
    assert len(key)>1 #cannot be the root directory
    tmp0 = len(key)
    ret = [x['key'][tmp0:].split('/',1)[0] for x in obs_client.listObjects(bucket_name, key)['body']['contents']]
    ret = sorted({x for x in ret if x})
    return ret
    
def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
    ```python
    >>> x = torch.arange(12).reshape(6, 2)
    >>> y = torch.arange(6).reshape(6, 1)
    >>> tensor_dict = {"x": x, "y": y}
    >>> split_tensor_dict(tensor_dict, 3)
    [
        {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
        {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
        {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
    ]
    ```
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if isinstance(tensor, torch.Tensor) else tensor
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]

def setup_fs_queue(base_path: str):
    """创建队列和处理目录"""
    queue_dir = Path(base_path) / "queue"
    processing_dir = Path(base_path) / "processing"
    queue_dir.mkdir(parents=True, exist_ok=True)
    processing_dir.mkdir(parents=True, exist_ok=True)
    return queue_dir, processing_dir


# @profiling_decorator
# def push_to_fs_queue(self, data: Dict[str, Any]):
#     """
#     将包含 PyTorch 张量的数据字典原子地写入文件队列。
#     使用 torch.save 进行序列化。
#     """
#     # 1. 写入临时文件。使用 .tmp 后缀以示区分。
#     # 使用 torch.save 保存数据字典。
#     # 注意：为了跨进程安全地加载，所有张量在保存前都应该在 CPU 上。
#     # (这个操作应该在调用此函数之前，在 sampler_script.py 中完成)
#     # queue_dir = self.queue_dir / f"{self.model_ids}/{self.rank}"
#     # self.queue_dir.mkdir(parents=True, exist_ok=True)
#     time_save = time.time()
#     learner_world_size = int(os.environ.get('LEARNER_WORLD_SIZE', 4))
#     assert data['advantages'].shape[0] % learner_world_size == 0, f"{data['advantages'].shape[0]} should be able to be divided by the number of GPUs that learner used, but now the number of GPUs that learner used is {learner_world_size}"
#     data_to_rank = split_tensor_dict(data, learner_world_size)
#     for rank in range(learner_world_size):
#         tmp_filename = f"tmp_{uuid.uuid4().hex}.pt"  # 使用 .pt 扩展名
#         tmp_path = self.queue_dir / tmp_filename

#         try:
#             torch.save(data_to_rank[rank], tmp_path)
#         except Exception as e:
#             print(f"ERROR: Failed to save data to temporary file {tmp_path}. Error: {e}")
#             # 如果保存失败，清理临时文件
#             if tmp_path.exists():
#                 tmp_path.unlink()
#             raise

#         assert self.model_ids == data["model_ids"]
#         # 2. 原子地重命名为正式文件，表示数据已准备好被消费。
#         # 这种方式可以防止消费者读到不完整的文件。
#         final_filename = f"data_{int(time_save * 1000)}_LearnerRank_{rank}_ModelID_{self.model_ids}_SamplerID_{self.sampler_id}_{uuid.uuid4().hex[:6]}.pt"
#         final_path = self.obs_queue_dir / final_filename
#         # wait_for_everyone()
#         # os.rename(tmp_path, final_path)
#         status = obs_upload_file(self.obs_client, str(tmp_path), str(final_path))
#         if status == 200:
#             print(f"status:{status}, file have been uploaded to: {final_path}")
#             os.remove(tmp_path)
#     print(f"rank_{self.rank} uploaded {learner_world_size} files")

@profiling_decorator
def push_to_fs_queue(self, data: Dict[str, Any]):
    """
    Atomically write a data dictionary containing PyTorch tensors to a file queue.
    Use torch.save for serialization.
    """
    time_save = time.time()
    learner_world_size = int(os.environ.get('LEARNER_WORLD_SIZE', 4))
    assert data['advantages'].shape[0] % learner_world_size == 0, f"{data['advantages'].shape[0]} should be able to be divided by the number of GPUs that learner used, but now the number of GPUs that learner used is {learner_world_size}"
    data_to_rank = split_tensor_dict(data, learner_world_size)

    tmp_files = []
    final_files = []
    
    try:
        for rank in range(learner_world_size):
            tmp_filename = f"tmp_{uuid.uuid4().hex}.pt"
            tmp_path = self.queue_dir / tmp_filename
            
            final_filename = f"data_{int(time_save * 1000)}_LearnerRank_{rank}_ModelID_{self.model_ids}_SamplerID_{self.sampler_id}_{uuid.uuid4().hex[:6]}.pt"
            final_path = self.obs_queue_dir + final_filename
            
            # 保存临时文件
            torch.save(data_to_rank[rank], tmp_path)
            
            tmp_files.append(tmp_path)
            final_files.append(final_path)
        
        # 2. 如果所有临时文件都创建成功，才进行原子重命名
        for tmp_path, final_path in zip(tmp_files, final_files):
            # wait_for_everyone()
            # os.rename(tmp_path, final_path)
            print(f"final_path:{str(final_path)}")
            status = obs_upload_file(self.obs_client, str(tmp_path), final_path)
            if status == 200:
                print(f"status:{status}, file have been uploaded to: {final_path}")
                os.remove(tmp_path)
        print(f"rank_{self.rank} uploaded {learner_world_size} files")
        
    except Exception as e:
        # 如果发生任何错误，清理所有已创建的文件
        print(f"ERROR: Failed to save data. Cleaning up all files. Error: {e}")
        
        # 清理临时文件
        for tmp_path in tmp_files:
            if tmp_path.exists():
                tmp_path.unlink()
                
        # 清理已重命名的最终文件
        for final_path in final_files:
            if final_path.exists():
                final_path.unlink()
                
        raise


def get_rank_from_name(name):
    return int(name.split("_")[3])


def get_model_id_from_name(name):
    return int(name.split("_")[5])


def get_max_model_id(sorted_name_list, rank: int):
    if len(sorted_name_list) == 0:
        return -1, None
    if isinstance(sorted_name_list[0], str):
        rank_pt_name_list = [file for file in sorted_name_list if get_rank_from_name(file) == rank]
        rank_model_ids_list = [get_model_id_from_name(file) for file in rank_pt_name_list]
    else:
        rank_pt_name_list = [file for file in sorted_name_list if get_rank_from_name(file.name) == rank]
        rank_model_ids_list = [get_model_id_from_name(file.name) for file in rank_pt_name_list]
    assert len(
        rank_model_ids_list) != 0, f"rank_model_ids_list is {rank_model_ids_list}, but sorted_name_list is {sorted_name_list}, rank_pt_name_list is {rank_pt_name_list}"
    max_value = max(rank_model_ids_list)
    max_index = rank_model_ids_list.index(max_value)  # 最大值的索引
    return max_value, rank_pt_name_list[max_index]


def get_min_model_id(sorted_name_list, min_id, rank: int):
    if len(sorted_name_list) == 0:
        return -1, None
    if isinstance(sorted_name_list[0], str):
        rank_pt_name_list = [file for file in sorted_name_list if get_rank_from_name(file) == rank]
        for file in rank_pt_name_list:
            valid_id = get_model_id_from_name(file)
            if valid_id >= min_id:
                return valid_id, file
    else:
        rank_pt_name_list = [file for file in sorted_name_list if get_rank_from_name(file.name) == rank]
        for file in rank_pt_name_list:
            valid_id = get_model_id_from_name(file.name)
            if valid_id >= min_id:
                return valid_id, file


@profiling_decorator
def pop_from_fs_queue(self, queue_dir: Path, processing_dir: Path, rank: int, timeout: int = 600, AIS_len: int = 8,
                      max_diff_step: int = 12) -> Optional[Dict[str, Any]]:
    """
    原子地从文件队列中获取一个文件，使用 torch.load 读取，并返回其内容。
    这是一个阻塞式操作，为多进程消费者设计。
    """
    learner_model_id = self.state.global_step
    # print("async_utils.py line 163",self._metrics)

    last_train_model_id = self.last_model_id

    if rank == 0:
        print(f"\nlast_train_model_id:{last_train_model_id}, learner_model_id:{learner_model_id} \n")

    while True:
        sorted_queue_dir = sorted(obs_listdir(self.obs_client, self.obs_queue_dir+"data"))
        num_files_of_queue = len(sorted_queue_dir)
        if num_files_of_queue % self.args.world_size != 0:
            print(f"文件数{num_files_of_queue}不是{self.args.world_size}的倍数，跳过")
            time.sleep(1.0)
            continue
        wait_for_everyone()

        sorted_queue_dir = sorted_queue_dir[-256:]
        queue_dir_max_model_id, path_in_queue_max = get_max_model_id(sorted_queue_dir, rank)

        if not path_in_queue_max:
            time.sleep(1.0)  # 队列为空，短暂等待后重试
            # print(f"短暂等待后重试: queue_dir 为空\n")
            continue
        # 学习器id-采样器最新模型id > 最大延迟,  短暂等待后重试
        elif learner_model_id - queue_dir_max_model_id > max_diff_step:
            time.sleep(1.0)
            # print(f"短暂等待后重试: 学习器id({learner_model_id})-采样器最新模型id({queue_dir_max_model_id}) > 最大延迟({max_diff_step})\n")
            continue
        # 普通重要性采样
        else:
            sorted_processing_dir = sorted(list(Path(processing_dir).glob("data_*.pt")))
            num_files_of_processing = len(sorted_processing_dir)
            sorted_processing_dir = sorted_processing_dir[-256:]
            processing_max_model_id, _ = get_max_model_id(sorted_processing_dir, rank)

            # 情况-1 queue里面出现新的id的数据
            if processing_max_model_id < queue_dir_max_model_id:
                path_of_data_to_learn = path_in_queue_max
            # 情况-2 queue里面没有出现新的id的数据，优先去寻找滑动窗口内的最旧数据（而不是优先去学习最新id下没有学完的数据）
            else:
                # theory_min_id = max(learner_model_id - max_diff_step,0) # 此处应该是以global-step计算，而不是以 processing_max_model_id
                theory_min_id = max(processing_max_model_id - max_diff_step,
                                    0)  # 此处不是以global-step计算，而是以 processing_max_model_id
                # 寻找滑窗内的最旧数据
                queue_dir_min_model_id, path_in_queue_min = get_min_model_id(sorted_queue_dir, theory_min_id, rank)
                # 没有最新（指新的id）数据到达时，永远学习滑窗内最旧的数据
                path_of_data_to_learn = path_in_queue_min

            # data_to_learn = torch.load(path_of_data_to_learn, map_location='cpu', weights_only=False)
            # wait_for_everyone()
            # os.rename(queue_dir / path_of_data_to_learn.name, processing_dir / path_of_data_to_learn.name)
            if isinstance(path_of_data_to_learn, str):
                path_of_data_to_learn_name = path_of_data_to_learn
            else:
                path_of_data_to_learn_name = path_of_data_to_learn.name
            obs_download_file(self.obs_client, self.obs_queue_dir + path_of_data_to_learn_name, str(queue_dir / path_of_data_to_learn_name))
            self.obs_client.deleteObject(self.obs_bucket_name, obs_path_to_key(self.obs_queue_dir + path_of_data_to_learn_name))
            data_to_learn = torch.load(str(queue_dir / path_of_data_to_learn_name), map_location='cpu', weights_only=False)
            # wait_for_everyone()
            os.rename(queue_dir / path_of_data_to_learn_name, processing_dir / path_of_data_to_learn_name)


            data_to_learn['metrics']['train']['num_files_of_queue'] = [num_files_of_queue]
            data_to_learn['metrics']['train']['num_files_of_prcessing'] = [num_files_of_processing]
            data_to_learn['metrics']['train']['queue_dir_max_model_id'] = [queue_dir_max_model_id]
            data_to_learn['metrics']['train']['processed_max_model_id'] = [processing_max_model_id]
            return data_to_learn


# =================================================================================
# 2. 模型同步回调 (与之前相同)
# =================================================================================
# async_utils.py

class SamplerSyncCallback(TrainerCallback):
    """
    一个回调，用于在训练步骤结束时，定期将学习器的模型权重同步给采样器。
    """

    def __init__(self, trainer, sync_weights_path: Path, sync_steps: int):  # <--- 新增 trainer 参数
        self.trainer = trainer  # <--- 将 trainer 存为成员变量
        self.sync_weights_path = sync_weights_path
        self.sync_steps = sync_steps
        self.last_synced_step = -1

    def on_step_end(self, args: TrainingArguments, state, control, model: nn.Module, **kwargs):
        """
        在每个梯度更新步骤的末尾被调用。
        """
        if state.global_step > self.last_synced_step and state.global_step % self.sync_steps == 0:
            self.last_synced_step = state.global_step
            if state.is_world_process_zero:
                unwrapped_model = self.trainer.accelerator.unwrap_model(model)
                temp_path = self.sync_weights_path.with_suffix(".tmp")
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save((control.should_save, state.global_step, unwrapped_model.state_dict()),
                           temp_path)  # d20250717修改
                # os.rename(temp_path, self.sync_weights_path)
                status = obs_upload_file(self.trainer.obs_client, temp_path, self.trainer.obs_sync_weights_path)
                print(f"status:{status}, [Learner] Step {state.global_step}: Synced weights for sampler at {self.trainer.obs_sync_weights_path}")
