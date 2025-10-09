export TZ='Asia/Shanghai'
formatted_time=$(date "+%Y%m%d-%H-%M-%S")
########################## parameters ##########################
scriptname=$1
xth=$2
cfg=$3
loss_type=$4
wandb_name=$5
sampler_id=$6
########################## parameters ##########################
cp /code/tmp/debug/torch_checkpoint_engine.py /opt/conda/envs/openrlhf/lib/python3.10/site-packages/deepspeed/runtime/checkpoint_engine/torch_checkpoint_engine.py
echo "weights_only=False"

log_path=/userhome/Research_HUB/GPG/hetero_rl/log_dir/sampler/${loss_type}/$1_sampler${sampler_id}_$2_cfg$3_${formatted_time}.log
mkdir -p "$(dirname "$log_path")"
echo $log_path
export WANDB_MODE=offline
export WANDB_DIR=/userhome/Research_HUB/GPG/hetero_rl/wandb/sampler${sampler_id}
export USE_FLASH_ATTN=true
export PYTHONPATH=/userhome/Research_HUB/GPG/hetero_rl/src
export WORLD_SIZE=1
export RANK=0
export GPUS=8
export MASTER_ADDR="localhost"
export SAVEPATH=/extrahome0/save_dir/8gpus/Sampler_${xth}_cfg${cfg}/sampler${sampler_id}/Qwen3-8B
export FS_QUEUE_PATH=/extrahome0/save_dir/8gpus/Async_${xth}_cfg${cfg}/Rollout/Qwen3-8B
export SYNC_WEIGHTS_PATH=/extrahome0/save_dir/8gpus/Async_${xth}_cfg${cfg}/tmp/Qwen3-8B/async_checkpoint.pt
export SYNC_SAMPLER_STEPS=1
export MASTER_PORT=29521
vllm_gpu_memory_utilization=0.8
if ! [[ "$sampler_id" =~ ^[0-3]$ ]]; then
  echo "Error: sampler_id must be 0, 1, 2 or 3"
  exit 1
fi


accelerate launch --config_file recipes/accelerate_configs/ddp_8gpus.yaml \
  --num_machines $WORLD_SIZE --machine_rank $RANK  --num_processes=$GPUS  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  src/open_r1/$scriptname.py --output_dir $SAVEPATH \
  --save_strategy "steps" --save_steps 100000  --save_total_limit  5 \
  --num_train_epochs 3 --gradient_accumulation_steps 8 --max_completion_length 4096 --max_prompt_length 768 \
  --scale_rewards False --eval_strategy 'no' \
  --model_name_or_path "/extrahome0/HF_models/Qwen/Qwen3-8B" \
  --dataset_name "/extrahome0/HF_datasets/hetero_rl/simplelr_qwen_level3to5" \
  --log_completions True --logging_steps 32 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --num_generations 8 \
  --wandb_entity "pcl-zh" --wandb_project "GPG"  --report_to "wandb" \
  --config recipes/HeteroRL/config_$3.yaml \
  --num_samplers 1 --sampler_id $sampler_id \
  --wandb_name $wandb_name \
  --loss_type $loss_type \
  --resume_from_checkpoint False \
  --use_think True \
  --vllm_gpu_memory_utilization $vllm_gpu_memory_utilization > $log_path 2>&1

# bash sh_dir/Sampler_8gpus_single_benchmark_checkpoint.sh sampler_script_checkpoint 1L2S_BNPO_diff32_think_8B 1L1S bnpo 1L2S_BNPO_diff32_think_8B 0 &

