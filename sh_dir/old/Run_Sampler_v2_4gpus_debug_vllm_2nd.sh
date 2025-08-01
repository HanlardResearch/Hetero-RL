export WANDB_MODE=offline
export WANDB_DIR=/userhome/Research_HUB/GPG/open-r1/wandb/sampler
export PYTHONPATH=/userhome/Research_HUB/GPG/open-r1/src
export WORLD_SIZE=1
export RANK=0
export GPUS=4
export MASTER_ADDR="localhost"
export MASTER_PORT=29526
export SAVEPATH="/extrahome0/save_dir/GPG/4gpus/SamlperV2_debug_2nd/Qwen3-1.7B"
export FS_QUEUE_PATH="/extrahome0/save_dir/GPG/4gpus/AsyncV2_debug/Rollout/Qwen3-1.7B"
export SYNC_WEIGHTS_PATH="/extrahome0/save_dir/GPG/4gpus/AsyncV2/tmp_debug/Qwen3-1.7B/gpg_async_weights.pt"
export SYNC_SAMPLER_STEPS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
#rm $SYNC_WEIGHTS_PATH
#echo "rm$SYNC_WEIGHTS_PATH"
accelerate launch --config_file recipes/accelerate_configs/ddp_4gpus.yaml \
  --num_machines $WORLD_SIZE --machine_rank $RANK  --num_processes=$GPUS  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  src/open_r1/sampler_script_v2_vllm.py --config   recipes/Qwen2.5-Math-7B/grpo/config_simple_rl_math_l35_v1_vllm.yaml --output_dir $SAVEPATH \
  --save_strategy "steps" --save_steps 100000  --save_total_limit  5 \
  --num_train_epochs 5 --gradient_accumulation_steps 1 --max_completion_length 2048 --max_prompt_length 768 \
  --scale_rewards False --adjust_gd --min_inverse_alpha 0.5 --eval_strategy 'no' \
  --model_name_or_path "/extrahome0/HF_models/Qwen/Qwen3-1.7B" --dataset_name "/extrahome0/HF_datasets/open-r1/simplelr_qwen_level3to5" \
  --per_device_train_batch_size 8 --log_completions True \
  --wandb_entity "pcl-zh" --wandb_project "GPG"  --report_to "wandb" --vllm_gpu_memory_utilization 0.9\
