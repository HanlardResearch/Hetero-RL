
MODEL="/extrahome0/HF_models/Qwen/Qwen3-1.7B"

BASE_MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

task="math_500"
echo "Evaluating task: $task"
output_dir="/userhome/Research_HUB/GPG/open-r1/src/open_r1/lighteval_results"

lighteval vllm "$model_args" "custom|$task|0|0" \
    --custom-tasks /userhome/Research_HUB/GPG/open-r1/src/open_r1/evaluate_math.py \
    --use-chat-template \
    --tasks_loading_mode MINIMAL \
    --output-dir "$output_dir"