formatted_time=$(date "+%Y%m%d-%H-%M-%S")
scriptname=$1
xth=$2
export SYNC_SAMPLER_STEPS=$3

log_path=/userhome/Research_HUB/GPG/open-r1/log_dir/learner/$1_$2thExp_SyncF$3_${formatted_time}.log


export WANDB_MODE=offline
export WANDB_DIR=/userhome/Research_HUB/GPG/open-r1/wandb/learner
export PYTHONPATH=/userhome/Research_HUB/GPG/open-r1/src
export WORLD_SIZE=1
export RANK=0
export GPUS=4
export MASTER_ADDR="localhost"
export MASTER_PORT=29510
export SAVEPATH=/extrahome0/save_dir/GPG/4gpus/LearnerV3a_vllm_${xth}th/Qwen3-1.7B
export FS_QUEUE_PATH=/extrahome0/save_dir/GPG/4gpus/AsyncV3a_vllm_${xth}th/Rollout/Qwen3-1.7B
export SYNC_WEIGHTS_PATH=/extrahome0/save_dir/GPG/4gpus/AsyncV3a_vllm/tmp_${xth}th/Qwen3-1.7B/gpg_async_weights.pt


echo $log_path
#export CUDA_VISIBLE_DEVICES=4,5,6,7
rm -r $FS_QUEUE_PATH
rm $SYNC_WEIGHTS_PATH
accelerate launch --config_file recipes/accelerate_configs/zero2_4A100s.yaml \
  --num_machines $WORLD_SIZE --machine_rank $RANK  --num_processes=$GPUS  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  src/open_r1/$scriptname.py --output_dir $SAVEPATH \
  --num_train_epochs 5 --max_completion_length 2048 --max_prompt_length 768 \
  --scale_rewards False --adjust_gd --min_inverse_alpha 0.5 \
  --model_name_or_path "/extrahome0/HF_models/Qwen/Qwen3-1.7B" --dataset_name "/extrahome0/HF_datasets/open-r1/simplelr_qwen_level3to5" \
  --max_steps 1295 --per_device_train_batch_size 8  --gradient_accumulation_steps 8 \
  --save_strategy "steps" --save_steps 32  --save_total_limit  5 \
  --eval_strategy 'steps' --eval_steps 32 \
  --wandb_entity "pcl-zh" --wandb_project "GPG"  --report_to "wandb" \
  --eval_on_start True --log_completions True --logging_steps 1 \
  --config recipes/AsyncGPG/config_simple_rl_math_l35_nRMs_v1.yaml \
  --vllm_gpu_memory_utilization 0.25  > $log_path 2>&1


