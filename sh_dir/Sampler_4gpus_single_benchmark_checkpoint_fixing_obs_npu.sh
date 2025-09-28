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
# cp /code/tmp/debug/torch_checkpoint_engine.py /opt/conda/envs/openrlhf/lib/python3.10/site-packages/deepspeed/runtime/checkpoint_engine/torch_checkpoint_engine.py
# echo "weights_only=False"
log_path=/home/ma-user/work/HeteroRL/log_dir/sampler/${loss_type}/$1_sampler${sampler_id}_$2_cfg$3_${formatted_time}.log
mkdir -p "$(dirname "$log_path")"
echo $log_path
export WANDB_MODE=offline
export WANDB_DIR=/home/ma-user/work/HeteroRL/wandb/sampler${sampler_id}
export USE_FLASH_ATTN=false
export PYTHONPATH=/home/ma-user/work/HeteroRL/src
export WORLD_SIZE=1
export LEARNER_WORLD_SIZE=4
export RANK=0
export GPUS=1
export MASTER_ADDR="localhost"
export SAVEPATH=/home/ma-user/work/save_dir/4gpus/Sampler_${xth}_cfg${cfg}/sampler${sampler_id}/Qwen3-1.7B
export FS_QUEUE_PATH=/home/ma-user/work/save_dir/4gpus/Async_${xth}_cfg${cfg}/Rollout/Qwen3-1.7B
export SYNC_WEIGHTS_PATH=/home/ma-user/work/save_dir/4gpus/Async_${xth}_cfg${cfg}/tmp/Qwen3-1.7B/async_checkpoint.pt
export SYNC_SAMPLER_STEPS=1
bash sh_dir/setup_obs.sh
vllm_gpu_memory_utilization=0.3
# if ! [[ "$sampler_id" =~ ^[0-3]$ ]]; then
#   echo "Error: sampler_id must be 0, 1, 2 or 3"
#   exit 1
# fi
# export ASCEND_RT_VISIBLE_DEVICES=0
# export MASTER_PORT=29521
# vllm_gpu_memory_utilization=0.3
# if [[ $sampler_id -eq 0 ]]; then
#   export CUDA_VISIBLE_DEVICES="2,3,4,5"
#   export MASTER_PORT=29521
#   vllm_gpu_memory_utilization=0.3
# elif [[ $sampler_id -eq 1 ]]; then
#   export CUDA_VISIBLE_DEVICES="4,5,6,7"
#   export MASTER_PORT=29522
#   vllm_gpu_memory_utilization=0.6
#elif [[ $sampler_id -eq 2 ]]; then
#  export CUDA_VISIBLE_DEVICES="4,5,6,7"
#  export MASTER_PORT=29523
#  vllm_gpu_memory_utilization=0.3
#elif [[ $sampler_id -eq 3 ]]; then
#  export CUDA_VISIBLE_DEVICES="4,5,6,7"
#  export MASTER_PORT=29524
#  vllm_gpu_memory_utilization=0.6
# fi

#rm $SYNC_WEIGHTS_PATH
#echo "rm$SYNC_WEIGHTS_PATH"
# "/extrahome0/HF_models/Qwen/Qwen3-1.7B" \
accelerate launch --config_file recipes/accelerate_configs/ddp_4gpus_npu.yaml \
  --num_machines $WORLD_SIZE --machine_rank $RANK  --num_processes=$GPUS  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  src/open_r1/$scriptname.py --output_dir $SAVEPATH \
  --save_strategy "steps" --save_steps 100000  --save_total_limit  5 \
  --num_train_epochs 3 --gradient_accumulation_steps 8 --max_completion_length 2048 --max_prompt_length 768 \
  --scale_rewards False --eval_strategy 'no' \
  --model_name_or_path "/home/ma-user/work/model/Qwen3-1.7B" \
  --dataset_name "/home/ma-user/work/dataset/simplelr_qwen_level3to5" \
  --log_completions True --logging_steps 32 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_generations 8 \
  --wandb_entity "pcl-zh" --wandb_project "GPG"  --report_to "wandb" \
  --config recipes/AsyncGRPO/config_simple_rl_math_l35_nRMs_${cfg}_npu.yaml \
  --num_samplers 20 --sampler_id $sampler_id \
  --wandb_name $wandb_name \
  --loss_type $loss_type \
  --resume_from_checkpoint False \
  --use_vllm False --use_think False \
  --vllm_gpu_memory_utilization $vllm_gpu_memory_utilization > $log_path 2>&1 &