# sleep 7200s
export TZ='Asia/Shanghai'
formatted_time=$(date "+%Y%m%d-%H-%M-%S")
loss_type=$1
wandb_name=$2
model_name=$3
log_path=/userhome/Research_HUB/GPG/open-r1/log_dir/rebuttal/online/${loss_type}/${wandb_name}_${formatted_time}.log
mkdir -p "$(dirname "$log_path")"
export WANDB_MODE=offline
export WANDB_DIR=/userhome/Research_HUB/GPG/open-r1/wandb/rebuttal/online/${loss_type}
export USE_FLASH_ATTN=true
export PYTHONPATH=/userhome/Research_HUB/GPG/open-r1/src
export WORLD_SIZE=1
export RANK=0
export GPUS=4
export MASTER_ADDR="localhost"
export SAVEPATH="/userhome/save_dir/4gpus/online/${loss_type}/${formatted_time}/${model_name}"

# data model config
#export dataset_name="/userhome/Research_HUB/GPG/open-r1/Data_Dir/deepmath_103k_dataset"
export dataset_name="/extrahome0/HF_datasets/open-r1/simplelr_qwen_level3to5"
#export model_name_or_path="/extrahome0/HF_models/DeepSeek-R1-Distill-Qwen-1.5B"
export model_name_or_path="/extrahome0/HF_models/${model_name}"
export config="recipes/OnlineRL/config_v1.yaml"

echo $log_path
accelerate launch --config_file recipes/accelerate_configs/zero2_4A100s.yaml \
  --num_machines $WORLD_SIZE --machine_rank $RANK  --num_processes=$GPUS  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  src/open_r1/online_rl.py --config $config --output_dir $SAVEPATH \
  --save_total_limit 1 --num_train_epochs 5 --gradient_accumulation_steps 8 --max_completion_length 2048 --max_prompt_length 768 \
  --scale_rewards False --model_name_or_path $model_name_or_path --dataset_name $dataset_name \
  --save_strategy "steps" --save_steps 256  --log_completions False --top_p $4 --top_k $5 \
  --temperature $6 --wandb_entity "pcl-zh"  --wandb_project "GPG"  --report_to "wandb"   \
  --per_device_eval_batch_size 8  --per_device_train_batch_size 8 --eval_strategy "steps" --eval_steps 128 --eval_on_start False --use_benchmark \
  --logging_steps 1  --use_vllm True --loss_type $loss_type  --wandb_name $wandb_name > $log_path 2>&1 &

