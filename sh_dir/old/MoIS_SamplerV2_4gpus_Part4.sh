formatted_time=$(date "+%Y%m%d-%H-%M-%S")
########################## parameters ##########################
scriptname=$1
xth=$2
cfg=$3
########################## parameters ##########################
log_path=/userhome/Research_HUB/GPG/open-r1/log_dir/AsyncGRPO/sampler/$1_part1_$2_cfg$3_${formatted_time}.log
echo $log_path
export WANDB_MODE=offline
export WANDB_DIR=/userhome/Research_HUB/GPG/open-r1/wandb/AsyncGRPO/sampler
export PYTHONPATH=/userhome/Research_HUB/GPG/open-r1/src
export WORLD_SIZE=1
export RANK=0
export GPUS=4
export MASTER_ADDR="localhost"
export MASTER_PORT=29524

export SAVEPATH=/userhome/save_dir/AsyncGRPO/4gpus/Sampler_${xth}_cfg${cfg}/part1/Qwen3-1.7B
export FS_QUEUE_PATH=/userhome/save_dir/AsyncGRPO/4gpus/Async_${xth}_cfg${cfg}/Rollout/Qwen3-1.7B
export SYNC_WEIGHTS_PATH=/userhome/save_dir/AsyncGRPO/4gpus/Async_${xth}_cfg${cfg}/tmp/Qwen3-1.7B/gpg_async_weights.pt

export SYNC_SAMPLER_STEPS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
#rm $SYNC_WEIGHTS_PATH
#echo "rm$SYNC_WEIGHTS_PATH"
accelerate launch --config_file recipes/accelerate_configs/ddp_4gpus.yaml \
  --num_machines $WORLD_SIZE --machine_rank $RANK  --num_processes=$GPUS  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  src/open_r1/$scriptname.py --output_dir $SAVEPATH \
  --save_strategy "steps" --save_steps 100000  --save_total_limit  5 \
  --num_train_epochs 3 --gradient_accumulation_steps 1 --max_completion_length 2048 --max_prompt_length 768 \
  --scale_rewards False --eval_strategy 'no' \
  --model_name_or_path "/extrahome0/HF_models/Qwen/Qwen3-1.7B" \
  --dataset_name "/extrahome0/HF_datasets/open-r1/simplelr_qwen_level3to5" \
  --log_completions False --logging_steps 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_generations 8 \
  --wandb_entity "pcl-zh" --wandb_project "GPG"  --report_to "wandb" \
  --config recipes/AsyncGRPO/config_simple_rl_math_l35_nRMs_$3.yaml \
  --sampler_id 3 \
  --vllm_gpu_memory_utilization 0.90 > $log_path 2>&1
