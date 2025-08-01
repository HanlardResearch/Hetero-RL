formatted_time=$(date "+%Y%m%d-%H-%M-%S")
log_path=/userhome/Research_HUB/GPG/open-r1/log_dir/grpo/${formatted_time}.log

export WANDB_MODE=offline
export WANDB_DIR=/userhome/Research_HUB/GPG/open-r1/wandb/grpo
export USE_FLASH_ATTN=true
export PYTHONPATH=/userhome/Research_HUB/GPG/open-r1/src
export WORLD_SIZE=1
export RANK=0
export GPUS=4
export MASTER_ADDR="localhost"
export MASTER_PORT=29506
export SAVEPATH="/extrahome0/save_dir/GPG/4gpus/GRPO/Qwen3-1.7B"
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo $log_path
accelerate launch --config_file recipes/accelerate_configs/zero2_4A100s.yaml \
  --num_machines $WORLD_SIZE --machine_rank $RANK  --num_processes=$GPUS  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  src/open_r1/grpo.py --config  recipes/Qwen2.5-Math-7B/grpo/config_simple_rl_math_l35_v1_vllm.yaml --output_dir $SAVEPATH \
  --save_total_limit  5 --num_train_epochs 5 --gradient_accumulation_steps 16 --max_completion_length 3072 --max_prompt_length 768 \
  --scale_rewards False --model_name_or_path "/extrahome0/HF_models/Qwen/Qwen3-1.7B" --dataset_name "/extrahome0/HF_datasets/open-r1/simplelr_qwen_level3to5" \
  --save_strategy "steps" --save_steps 32000 \
  --wandb_entity "pcl-zh"  --wandb_project "GPG"  --report_to "wandb"   \
  --per_device_eval_batch_size 16  --per_device_train_batch_size 4 --eval_strategy "steps" --eval_steps 32 --eval_on_start True --seed 2025 \
  --logging_steps 1  --use_vllm True > $log_path 2>&1