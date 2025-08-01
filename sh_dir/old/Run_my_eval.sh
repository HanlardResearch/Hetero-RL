lighteval \
    --model_args pretrained=/extrahome0/HF_models/Qwen/Qwen3-1.7B,dtype=bfloat16 \
    --tasks math_500 \
    --custom_tasks /userhome/Research_HUB/GPG/open-r1/src/open_r1/evaluate_short.py \
    --override_batch_size 8 \
    --output_dir ./results/Qwen3-1.7B-on-math500