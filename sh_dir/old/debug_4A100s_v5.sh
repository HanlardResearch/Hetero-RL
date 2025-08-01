export WANDB_MODE=offline
export WANDB_DIR=/userhome/Research_HUB/GPG/open-r1/wandb/debug
export USE_FLASH_ATTN=true
export PYTHONPATH=/userhome/Research_HUB/GPG/open-r1/src
export WORLD_SIZE=1
export RANK=0
export GPUS=4
export MASTER_ADDR="localhost"
export MASTER_PORT=29504
export SAVEPATH="/extrahome0/save_dir/GPG/4gpus/debug/Qwen3-1.7B"
accelerate launch --config_file recipes/accelerate_configs/zero2_4A100s.yaml \
  --num_machines $WORLD_SIZE --machine_rank $RANK  --num_processes=$GPUS  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  src/open_r1/gpg.py --config  recipes/Qwen2.5-Math-7B/grpo/config_simple_rl_math_l35_v1.yaml --output_dir $SAVEPATH \
  --save_total_limit  5 --num_train_epochs 5 --gradient_accumulation_steps 1 --max_completion_length 2048 --max_prompt_length 768 \
  --scale_rewards False --adjust_gd --min_inverse_alpha 0.5 --eval_strategy epoch \
  --model_name_or_path "/extrahome0/HF_models/Qwen/Qwen3-1.7B" --dataset_name "/extrahome0/HF_datasets/open-r1/simplelr_qwen_level3to5" \
  --save_strategy "steps" --save_steps 32 \
  --per_device_train_batch_size 8  --eval_strategy 'steps' --eval_steps 1 --eval_on_start True \