import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from transformers import TrainerCallback, TrainingArguments
from typing import Optional, Dict, Any
import os

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


def push_to_fs_queue(queue_dir: Path, data: Dict[str, Any]):
    """
    将包含 PyTorch 张量的数据字典原子地写入文件队列。
    使用 torch.save 进行序列化。
    """
    # 1. 写入临时文件。使用 .tmp 后缀以示区分。
    # 使用 torch.save 保存数据字典。
    # 注意：为了跨进程安全地加载，所有张量在保存前都应该在 CPU 上。
    # (这个操作应该在调用此函数之前，在 sampler_script.py 中完成)
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
    final_filename = f"data_{int(time.time_ns())}_{uuid.uuid4().hex[:6]}.pt"
    final_path = queue_dir / final_filename
    os.rename(tmp_path, final_path)
    print(f"文件保存在: {final_path}")


def pop_from_fs_queue(queue_dir: Path, processing_dir: Path, rank: int, timeout: int = 600) -> Optional[Dict[str, Any]]:
    """
    原子地从文件队列中获取一个文件，使用 torch.load 读取，并返回其内容。
    这是一个阻塞式操作，为多进程消费者设计。
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        # 1. 查找队列中的所有数据文件
        # 使用 glob 匹配正式文件名模式
        files = sorted(list(queue_dir.glob("data_*.pt")))
        if not files:
            # print(f"{queue_dir}空了，等待0.1秒")
            time.sleep(0.1)  # 队列为空，短暂等待后重试
            continue

        # 2. 尝试获取第一个文件
        source_path = files[0]

        # 3. 【核心】通过原子重命名操作来“锁定”文件
        # 将文件从 'queue' 目录移动到 'processing' 目录，并加上 rank 标记。
        # 只有一个进程能成功，其他进程会因为 FileNotFoundError 而进入下一轮循环。
        processing_filename = f"proc_{rank}_{source_path.name}"
        processing_path = processing_dir / processing_filename

        try:
            os.rename(source_path, processing_path)
            # print(f"文件锁定: \n"
            #       f"{source_path} \n"
            #       f"--->\n"
            #       f"{processing_path}")
        except FileNotFoundError:
            # 文件被其他进程抢先了，这是正常的多进程竞争，继续循环即可。
            continue

        # 4. 只有成功“锁定”文件的进程会执行到这里
        try:
            # 使用 torch.load 读取数据。
            # map_location='cpu' 是一个好习惯，确保数据加载到 CPU内存，
            # 之后再由每个进程自己决定是否以及何时移动到特定的 GPU。
            data = torch.load(processing_path,  map_location='cpu', weights_only=False)
            return data
        except Exception as e:
            print(f"ERROR [Rank {rank}]: Failed to load data from {processing_path}. Error: {e}")
            # 即使加载失败，也要继续执行 finally 块来清理文件
            # 返回 None 或重新抛出异常，取决于你希望的行为
            return None  # 或者 raise e
        finally:
            # 5. 确保在处理完成后（无论成功还是失败）都删除文件，避免处理目录堆积
            if processing_path.exists():
                processing_path.unlink()

    # 如果在指定时间内没有等到任何文件，则超时
    print(f"WARNING [Rank {rank}]: Timed out after {timeout}s waiting for data from the file queue at '{queue_dir}'.")
    return None

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

                # 【关键修改】在保存之前，确保临时文件的父目录存在
                # temp_path.parent 获取父目录的 Path 对象
                # .mkdir(parents=True, exist_ok=True) 会：
                #   - parents=True: 如果需要，会一并创建所有上层父目录
                #   - exist_ok=True: 如果目录已经存在，不会报错
                temp_path.parent.mkdir(parents=True, exist_ok=True)

                # 现在可以安全地保存了
                torch.save(unwrapped_model.state_dict(), temp_path)

                os.rename(temp_path, self.sync_weights_path)

                print(f"[Learner] Step {state.global_step}: Synced weights for sampler at {self.sync_weights_path}")

        # if state.global_step > self.last_synced_step and state.global_step % self.sync_steps == 0:
        #     self.last_synced_step = state.global_step
        #
        #     # Accelerator 会处理好进程同步，但我们通常还是在主进程执行IO操作
        #     if state.is_world_process_zero:
        #         print(f"[Learner] Step {state.global_step}: Gathering and syncing weights for sampler...")
        #
        #     # 【关键修改】使用 accelerator 来获取完整的 state_dict
        #     # 这个函数是阻塞的，它会等待所有进程的参数都聚合过来
        #     # 默认情况下，它会将权重聚合到 CPU 以避免 OOM
        #     full_state_dict = self.trainer.accelerator.get_state_dict(model)
        #
        #     # 只有主进程负责写入文件
        #     if state.is_world_process_zero:
        #         temp_path = self.sync_weights_path.with_suffix(".tmp")
        #         temp_path.parent.mkdir(parents=True, exist_ok=True)
        #
        #         # 保存的是聚合后的完整 state_dict
        #         torch.save(full_state_dict, temp_path)
        #
        #         os.rename(temp_path, self.sync_weights_path)
        #         print(f"[Learner] Step {state.global_step}: Synced weights for sampler at {self.sync_weights_path}")