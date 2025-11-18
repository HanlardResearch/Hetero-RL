export TZ='Asia/Shanghai'
formatted_time=$(date "+%Y%m%d-%H-%M-%S")
########################## parameters ##########################
scriptname=$1
xth=$2
export SYNC_SAMPLER_STEPS=$3
cfg=$4
loss_type=$5
wandb_name=$6
ModelPath="/extrahome0/HF_models/Qwen/Qwen3-1.7B"
########################## parameters ##########################


log_path=/userhome/Research_HUB/GPG/Hetero-RL/log_dir/rebuttal/learner/${loss_type}/${wandb_name}_$1_$2_SyncF$3_cfg${cfg}_${formatted_time}.log
mkdir -p "$(dirname "$log_path")"

export WANDB_MODE=offline
export WANDB_DIR=/userhome/Research_HUB/GPG/Hetero-RL/wandb/rebuttal/learner/${loss_type}
export USE_FLASH_ATTN=true
export PYTHONPATH=/userhome/Research_HUB/GPG/Hetero-RL/src
export WORLD_SIZE=1
export RANK=0
export GPUS=4
export MASTER_ADDR="localhost"
export MASTER_PORT=29510
export SAVEPATH=/userhome/save_dir/4gpus/Learner_${xth}_cfg${cfg}/Qwen3-1.7B
export FS_QUEUE_PATH=/userhome/save_dir/4gpus/Async_${xth}_cfg${cfg}/Rollout/Qwen3-1.7B
export SYNC_WEIGHTS_PATH=/userhome/save_dir/4gpus/Async_${xth}_cfg${cfg}/tmp/Qwen3-1.7B/async_checkpoint.pt
export QUEUE_TIMEOUT_SECONDS=3600

echo $log_path
# export CUDA_VISIBLE_DEVICES=1,2,3,4
rm -r $FS_QUEUE_PATH
rm $SYNC_WEIGHTS_PATH
accelerate launch --config_file recipes/accelerate_configs/zero2_4A100s.yaml \
  --num_machines $WORLD_SIZE --machine_rank $RANK  --num_processes=$GPUS  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  src/hetero_rl/$scriptname.py --output_dir $SAVEPATH \
  --max_prompt_length 768 \
  --model_name_or_path $ModelPath \
  --dataset_name "/extrahome0/HF_datasets/Hetero-RL/simplelr_qwen_level3to5" \
  --max_steps 1295 \
  --save_strategy "steps" --save_steps 64  --save_total_limit  1 \
  --eval_strategy 'steps' --eval_steps 64 \
  --wandb_entity "pcl-zh" --wandb_project "GPG"  --report_to "wandb" \
  --log_completions False --logging_steps 1 \
  --config recipes/HeteroRL/config_${cfg}.yaml \
  --vllm_gpu_memory_utilization 0.25  \
  --max_completion_length 2048 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --num_generations $7 \
  --cppo_beta $8 \
  --wandb_name $wandb_name \
  --ais_beta 0.5 \
  --use_benchmark \
  --loss_type $loss_type \
  --resume_from_checkpoint False \
  --use_think True \
  --kl_temp 0.001 \
  --eval_on_start False \
  --scale_rewards False > $log_path 2>&1 &
