export TZ='Asia/Shanghai'
formatted_time=$(date "+%Y%m%d-%H-%M-%S")
loss_type=$1
log_path=/home/ma-user/work/AsyGPG/log_dir/online/${loss_type}/${formatted_time}.log
mkdir -p "$(dirname "$log_path")"
export WANDB_MODE=offline
export WANDB_DIR=/home/ma-user/work/AsyGPG/wandb/online/${loss_type}
export USE_FLASH_ATTN=false
export PYTHONPATH=/home/ma-user/work/AsyGPG/src
export WORLD_SIZE=1
export RANK=0
export GPUS=1
export MASTER_ADDR="localhost"
export MASTER_PORT=29516
export NODE_RANK=0
# export HCCL_IF_IP="127.0.0.1"
export SAVEPATH="/home/ma-user/work/save_dir/GPG/4gpus/${loss_type}/${formatted_time}/Qwen3-1.7B"
export ASCEND_RT_VISIBLE_DEVICES=3
# if [[ $loss_type == "gspo" ]]; then
#     export CUDA_VISIBLE_DEVICES="0,1,2,3"
#     export MASTER_PORT=29508
# elif [[ $loss_type == "EqQ" ]]; then
#     export CUDA_VISIBLE_DEVICES="4,5,6,7"
#     export MASTER_PORT=29507
# elif [[ $loss_type == "grpo" ]]; then
#     export CUDA_VISIBLE_DEVICES="0,1,2,3"
#     export MASTER_PORT=29506
# fi

# if [[ $loss_type == "dr_grpo" ]]; then
#     export CUDA_VISIBLE_DEVICES="0,1,2,3"
#     export MASTER_PORT=29508
# elif [[ $loss_type == "bnpo" ]]; then
#     export CUDA_VISIBLE_DEVICES="4,5,6,7"
#     export MASTER_PORT=29507
# fi
echo $log_path
accelerate launch --config_file recipes/accelerate_configs/zero2_4A100s_npu.yaml \
  --num_machines $WORLD_SIZE --machine_rank $RANK  --num_processes=$GPUS  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  src/open_r1/online_rl_npu.py --config  recipes/Qwen2.5-Math-7B/grpo/config_simple_rl_math_l35_v1_vllm_npu.yaml --output_dir $SAVEPATH \
  --save_total_limit 1 --num_train_epochs 5 --max_completion_length 2048 --max_prompt_length 768 \
  --scale_rewards False --model_name_or_path "/home/ma-user/work/model/Qwen3-1.7B" --dataset_name "/home/ma-user/work/dataset/simplelr_qwen_level3to5" \
  --save_strategy "steps" --save_steps 64 --log_completions True \
  --wandb_entity "pcl-zh"  --wandb_project "GPG"  --report_to "wandb"   \
  --per_device_eval_batch_size 8  --per_device_train_batch_size 8 --eval_strategy "steps" --eval_steps 64 --eval_on_start True \
  --logging_steps 1  --use_vllm False --loss_type $loss_type > $log_path 2>&1 &
