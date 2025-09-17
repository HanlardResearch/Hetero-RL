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


def setup_fs_queue(base_path: str):
    queue_dir = Path(base_path) / "queue"
    processing_dir = Path(base_path) / "processing"
    queue_dir.mkdir(parents=True, exist_ok=True)
    processing_dir.mkdir(parents=True, exist_ok=True)
    return queue_dir, processing_dir


@profiling_decorator
def push_to_fs_queue(self, data: Dict[str, Any], time_save):
    tmp_filename = f"tmp_{uuid.uuid4().hex}.pt"
    tmp_path = self.queue_dir / tmp_filename

    try:
        torch.save(data, tmp_path)
    except Exception as e:
        print(f"ERROR: Failed to save data to temporary file {tmp_path}. Error: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    assert self.model_ids == data["model_ids"]
    final_filename = f"data_{int(time_save * 1000)}_SamplerRank_{self.rank}_ModelID_{self.model_ids}_{uuid.uuid4().hex[:6]}.pt"
    final_path = self.queue_dir / final_filename
    wait_for_everyone()
    os.rename(tmp_path, final_path)


def get_rank_from_name(name):
    return int(name.split("_")[3])


def get_model_id_from_name(name):
    return int(name.split("_")[5])


def get_max_model_id(sorted_name_list, rank: int):
    if len(sorted_name_list) == 0:
        return -1, None
    rank_pt_name_list = [file for file in sorted_name_list if get_rank_from_name(file.name) == rank]
    rank_model_ids_list = [get_model_id_from_name(file.name) for file in rank_pt_name_list]
    assert len(
        rank_model_ids_list) != 0, f"rank_model_ids_list is {rank_model_ids_list}, but sorted_name_list is {sorted_name_list}, rank_pt_name_list is {rank_pt_name_list}"
    max_value = max(rank_model_ids_list)
    max_index = rank_model_ids_list.index(max_value)
    return max_value, rank_pt_name_list[max_index]


def get_min_model_id(sorted_name_list, min_id, rank: int):
    if len(sorted_name_list) == 0:
        return -1, None
    rank_pt_name_list = [file for file in sorted_name_list if get_rank_from_name(file.name) == rank]
    for file in rank_pt_name_list:
        valid_id = get_model_id_from_name(file.name)
        if valid_id >= min_id:
            return valid_id, file


@profiling_decorator
def pop_from_fs_queue(self, queue_dir: Path, processing_dir: Path, rank: int, timeout: int = 600, AIS_len: int = 8,
                      max_diff_step: int = 12, world_size: int = 4) -> Optional[Dict[str, Any]]:

    learner_model_id = self.state.global_step
    last_train_model_id = self.last_model_id

    if rank == 0:
        print(f"\nlast_train_model_id:{last_train_model_id}, learner_model_id:{learner_model_id} \n")

    while True:
        sorted_queue_dir = sorted(list(Path(queue_dir).glob("data_*.pt")))
        num_files_of_queue = len(sorted_queue_dir)
        if num_files_of_queue % world_size != 0:
            time.sleep(1.0)
            continue
        wait_for_everyone()

        sorted_queue_dir = sorted_queue_dir[-256:]
        queue_dir_max_model_id, path_in_queue_max = get_max_model_id(sorted_queue_dir, rank)

        if not path_in_queue_max:
            time.sleep(1.0)
            continue
        elif learner_model_id - queue_dir_max_model_id > max_diff_step:
            time.sleep(1.0)
            continue
        else:
            sorted_processing_dir = sorted(list(Path(processing_dir).glob("data_*.pt")))
            num_files_of_processing = len(sorted_processing_dir)
            sorted_processing_dir = sorted_processing_dir[-256:]
            processing_max_model_id, _ = get_max_model_id(sorted_processing_dir, rank)

            if processing_max_model_id < queue_dir_max_model_id:
                path_of_data_to_learn = path_in_queue_max
            else:
                theory_min_id = max(processing_max_model_id - max_diff_step,
                                    0)
                queue_dir_min_model_id, path_in_queue_min = get_min_model_id(sorted_queue_dir, theory_min_id, rank)
                path_of_data_to_learn = path_in_queue_min

            data_to_learn = torch.load(path_of_data_to_learn, map_location='cpu', weights_only=False)
            wait_for_everyone()
            os.rename(queue_dir / path_of_data_to_learn.name, processing_dir / path_of_data_to_learn.name)

            data_to_learn['metrics']['train']['num_files_of_queue'] = [num_files_of_queue]
            data_to_learn['metrics']['train']['num_files_of_prcessing'] = [num_files_of_processing]
            data_to_learn['metrics']['train']['queue_dir_max_model_id'] = [queue_dir_max_model_id]
            data_to_learn['metrics']['train']['processed_max_model_id'] = [processing_max_model_id]
            return data_to_learn



class SamplerSyncCallback(TrainerCallback):

    def __init__(self, trainer, sync_weights_path: Path, sync_steps: int):
        self.trainer = trainer
        self.sync_weights_path = sync_weights_path
        self.sync_steps = sync_steps
        self.last_synced_step = -1

    def on_step_end(self, args: TrainingArguments, state, control, model: nn.Module, **kwargs):

        if state.global_step > self.last_synced_step and state.global_step % self.sync_steps == 0:
            self.last_synced_step = state.global_step
            if state.is_world_process_zero:
                unwrapped_model = self.trainer.accelerator.unwrap_model(model)
                temp_path = self.sync_weights_path.with_suffix(".tmp")
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save((control.should_save, state.global_step, unwrapped_model.state_dict()),
                           temp_path)
                os.rename(temp_path, self.sync_weights_path)
                print(f"[Learner] Step {state.global_step}: Synced weights for sampler at {self.sync_weights_path}")
