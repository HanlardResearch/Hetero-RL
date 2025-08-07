import logging
import os
import pickle
from pathlib import Path

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

def setup_fs_queue(base_path: str):
    """创建队列和处理目录"""
    queue_dir = Path(base_path) / "queue"
    processing_dir = Path(base_path) / "processing"
    queue_dir.mkdir(parents=True, exist_ok=True)
    processing_dir.mkdir(parents=True, exist_ok=True)
    return queue_dir, processing_dir

@profiling_decorator
def push_to_fs_queue(self, data: Dict[str, Any], time_save):
    """
    将包含 PyTorch 张量的数据字典原子地写入文件队列。
    使用 torch.save 进行序列化。
    """
    # 1. 写入临时文件。使用 .tmp 后缀以示区分。
    # 使用 torch.save 保存数据字典。
    # 注意：为了跨进程安全地加载，所有张量在保存前都应该在 CPU 上。
    # (这个操作应该在调用此函数之前，在 sampler_script.py 中完成)
    queue_dir = self.queue_dir / f"{self.model_ids}/{self.rank}"
    queue_dir.mkdir(parents=True, exist_ok=True)
    
    tmp_filename = f"tmp_{uuid.uuid4().hex}.pt"  # 使用 .pt 扩展名
    tmp_path = queue_dir / tmp_filename

    try:
        torch.save(data, tmp_path)
    except Exception as e:
        print(f"ERROR: Failed to save data to temporary file {tmp_path}. Error: {e}")
        # 如果保存失败，清理临时文件
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    # 2. 原子地重命名为正式文件，表示数据已准备好被消费。
    # 这种方式可以防止消费者读到不完整的文件。
    final_filename = f"data_{int(time_save*1000)}_SamplerRank_{self.rank}_{uuid.uuid4().hex[:6]}.pt"
    final_path = queue_dir / final_filename
    os.rename(tmp_path, final_path)
    print(f"文件保存在: {final_path}")
#
# @profiling_decorator
# def pop_from_fs_queue(self, queue_dir: Path, processing_dir: Path, rank: int, timeout: int = 600) -> Optional[Dict[str, Any]]:
#     """
#     原子地从文件队列中获取一个文件，使用 torch.load 读取，并返回其内容。
#     这是一个阻塞式操作，为多进程消费者设计。
#     """
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         delta_time = time.time() - start_time
#         # 1. 查找队列中的所有数据文件
#         # 使用 glob 匹配正式文件名模式
#         files = sorted(list(queue_dir.glob("data_*.pt")))
#         if not files:
#             # print(f"{queue_dir}空了，等待0.1秒")
#             time.sleep(0.1)  # 队列为空，短暂等待后重试
#             continue
#         queue_dir = self.queue_dir
#         if not self.queue_dir.exists():
#             print(f"Directory {queue_dir} does not exist.")
#             return None
#         model_ids_dirs = [item for item in queue_dir.iterdir()
#                     if item.is_dir() and item.name.isdigit()]
#         sorted_dirs = sorted(model_ids_dirs, key=lambda x: int(x.name))
#         length_model_ids = len(sorted_dirs)
#
#         if length_model_ids >= K:
#
#             # 遍历 queue_dir 下的所有子目录
#             for item in model_ids_dirs:
#                 print(f"Processing directory: {item.name}")
#                 # 获取该目录下所有文件（不包括子子目录中的文件，如需递归可用 rglob）
#                 files = [f for f in item.iterdir() if f.is_file()]
#
#                 if not files:
#                     print(f"  No files found in {item}")
#                     continue
#
#                 # 按修改时间排序，取最新的
#                 latest_file = max(files, key=lambda f: f.stat().st_mtime)
#                 print(f"  Latest file: {latest_file.name}, Modified: {latest_file.stat().st_mtime}")
#
#                 # 读取最新文件内容
#                 try:
#                     with open(latest_file, 'r', encoding='utf-8') as f:
#                         content = f.read()
#                         print(f"  Content (first 200 chars): {content[:200]}")
#                         # 或做其他处理...
#                 except Exception as e:
#                     print(f"  Failed to read {latest_file}: {e}")
#             queue_dir = self.queue_dir / f"{self.model_ids}/{self.rank}"
#         else:
#             self.loss_type = "pg"
#
#         # 2. 尝试获取最旧一个文件
#         latest_model_ids_dir = sorted_dirs[-1]
#         source_dir = latest_model_ids_dir / str(self.rank)
#         files = sorted(list(source_dir.glob("data_*.pt")))
#         source_path = files[0]
#
#
#         # 3. 【核心】通过原子重命名操作来“锁定”文件
#         # 将文件从 'queue' 目录移动到 'processing' 目录，并加上 rank 标记。
#         # 只有一个进程能成功，其他进程会因为 FileNotFoundError 而进入下一轮循环。
#         processing_filename = f"LearnerRank_{rank}_{source_path.name}"
#         processing_path = processing_dir / f"{latest_model_ids_dir.name}/{rank}/{processing_filename}"
#         processing_path.parent.mkdir(parents=True, exist_ok=True)
#         try:
#             os.rename(source_path, processing_path)
#             # print(f"文件锁定: \n"
#             #       f"{source_path} \n"
#             #       f"--->\n"
#             #       f"{processing_path}")
#         except FileNotFoundError:
#             # 文件被其他进程抢先了，这是正常的多进程竞争，继续循环即可。
#             continue
#
#         # 4. 只有成功“锁定”文件的进程会执行到这里
#         try:
#             # 使用 torch.load 读取数据。
#             # map_location='cpu' 是一个好习惯，确保数据加载到 CPU内存，
#             # 之后再由每个进程自己决定是否以及何时移动到特定的 GPU。
#             data = torch.load(processing_path,  map_location='cpu', weights_only=False)
#             return data
#         except Exception as e:
#             print(f"ERROR [Rank {rank}]: Failed to load data from {processing_path}. Error: {e}")
#             # 即使加载失败，也要继续执行 finally 块来清理文件
#             # 返回 None 或重新抛出异常，取决于你希望的行为
#             return None  # 或者 raise e
#         finally:
#             # 5. 确保在处理完成后（无论成功还是失败）都删除文件，避免处理目录堆积
#             if processing_path.exists():
#             #     processing_path.unlink()
#                 print(f"文件迁移至：{processing_path}")
#
#     # 如果在指定时间内没有等到任何文件，则超时
#     print(f"WARNING [Rank {rank}]: Timed out after {delta_time} (Max-{timeout})s waiting for data from the file queue at '{queue_dir}'.")
#     return None


@profiling_decorator
def pop_from_fs_queue(self, queue_dir: Path, processing_dir: Path, rank: int, timeout: int = 600, AIS_len: int = 8, max_diff_step: int = 12) -> Optional[Dict[str, Any]]:
    """
    原子地从文件队列中获取一个文件，使用 torch.load 读取，并返回其内容。
    这是一个阻塞式操作，为多进程消费者设计。
    """
    learner_model_id = self.state.global_step
    # print("async_utils.py line 163",self._metrics)

    last_train_model_id = self.last_model_id

    print(f"last_train_model_id:{last_train_model_id}, learner_model_id:{learner_model_id}")
    while True:

        sampler_model_ids = sorted([int(model_id) for model_id in os.listdir(queue_dir)])
        # 1. 查找队列中的所有数据文件
        # 使用 glob 匹配正式文件名模式

        if not sampler_model_ids:
            time.sleep(3.0)  # 队列为空，短暂等待后重试
            print(f"短暂等待后重试: sampler_model_ids 为空")
            continue
        # 学习器id-采样器最新模型id > 最大延迟,  短暂等待后重试
        elif  learner_model_id - sampler_model_ids[-1] > max_diff_step:
            time.sleep(1.0)
            print(f"短暂等待后重试: 学习器id({learner_model_id})-采样器最新模型id({sampler_model_ids[-1]}) > 最大延迟({max_diff_step})")
            continue
        # 普通重要性采样
        elif len(sampler_model_ids) < AIS_len:
            if sampler_model_ids[-1] <= last_train_model_id:# 每次学习新id数据
                time.sleep(1.0)
                print(
                    f"短暂等待后重试(): sampler_model_ids[-1] <=last_train_model_id: {sampler_model_ids[-1]} <={last_train_model_id}")
                continue
            queue_dir_wt_id_rank = queue_dir / str(sampler_model_ids[-1]) / str(self.rank)
            files = sorted(list(queue_dir_wt_id_rank.glob("data_*.pt")))
            if not files:
                time.sleep(1.0)  # 队列为空，短暂等待后重试
                print(f"短暂等待后重试: 路径内没有.pt文件 {queue_dir_wt_id_rank} ")
                continue

            source_path = files[0]
            try:
                data = torch.load(source_path, map_location='cpu', weights_only=False)
                data['history_advs'] = torch.zeros([self.args.per_device_train_batch_size*self.args.gradient_accumulation_steps, AIS_len])
                print(f"【重要性采样数据】:{source_path}")
                return data
            except Exception as e:
                print(f"ERROR [Rank {self.rank}]: Failed to load data from {source_path}. Error: {e}")
                # 即使加载失败，也要继续执行 finally 块来清理文件
                # 返回 None 或重新抛出异常，取决于你希望的行为
                return None  # 或者 raise e
        # 模拟退火重要性采样
        else:
            # if sampler_model_ids[-1] < last_train_model_id-1:# 每次学习新id数据
            #     time.sleep(5.0)
            #     print(
            #         f"短暂等待后重试(): sampler_model_ids[-1] <=last_train_model_id: {sampler_model_ids[-1]} <={last_train_model_id}")
            #     continue
            history_advs = []
            for i in range(AIS_len,0,-1):
                queue_dir_wt_id_rank = queue_dir / str(sampler_model_ids[-i]) / str(self.rank)
                files = sorted(list(queue_dir_wt_id_rank.glob("data_*.pt")))
                if not files:
                    history_advs.append(torch.zeros([self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps, 1]))
                    continue
                # 随机 读 同一个mode_id 下面一个文件，保证各个rank读的是一个时间戳即可
                source_path = files[learner_model_id % len(files)]
                # 4. 只有成功“锁定”文件的进程会执行到这里
                try:
                    data = torch.load(source_path, map_location='cpu', weights_only=False)
                    # 每行对应梯度累计step
                    print(f"【退火重要性采样数据】:{source_path}")
                    if i > 1:
                        history_advs.append(data['advantages'].unsqueeze(1)) # [64, 1]
                    else:
                        # [num_generations* gradient_accumulation_steps, AIS_len]张量
                        data['history_advs'] = torch.cat(history_advs,dim=1)
                        return data
                except Exception as e:
                    print(f"ERROR [Rank {self.rank}]: Failed to load data from {source_path}. Error: {e}")
                    return None  # 或者 raise e

# =================================================================================
# 2. 模型同步回调 (与之前相同)
# =================================================================================
# async_utils.py

class SamplerSyncCallback(TrainerCallback):
    """
    一个回调，用于在训练步骤结束时，定期将学习器的模型权重同步给采样器。
    """
    def __init__(self, trainer, sync_weights_path: Path, sync_steps: int): # <--- 新增 trainer 参数
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
                torch.save((state.global_step, unwrapped_model.state_dict()), temp_path) # d20250717修改
                os.rename(temp_path, self.sync_weights_path)
                print(f"[Learner] Step {state.global_step}: Synced weights for sampler at {self.sync_weights_path}")
