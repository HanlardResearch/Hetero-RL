export PYTHONPATH=src

accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
  --num_machines $WORLD_SIZE --machine_rank $RANK  --num_processes=$GPUS  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  src/open_r1/gpg.py --config   recipes/Qwen2.5-Math-7B/grpo/config_simple_rl_math_l35_v1.yaml --output_dir  Your_Path \
  --save_strategy "epoch" --save_total_limit  5 --num_train_epochs 5 --gradient_accumulation_steps 4 --max_completion_length 2048 --max_prompt_length 768 \
  --scale_rewards False --adjust_gd --min_inverse_alpha 0.5 --eval_strategy "steps" \