formatted_time=$(date "+%Y%m%d-%H-%M-%S")
loss_type=$1
length=2048
export WANDB_MODE=offline
export WANDB_DIR=/userhome/Research_HUB/GPG/open-r1/wandb/grpo
export USE_FLASH_ATTN=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=/userhome/Research_HUB/GPG/open-r1/src
# export SAVEPATH="/extrahome0/save_dir/GPG/4gpus/${loss_type}/Qwen3-1.7B"
export SAVEPATH="/extrahome0/save_dir/GPG/4gpus/GRPO/Qwen3-1.7B/checkpoint-1295"
export NUM_GPUS=4 # Set to 8 for 32B and 70B models
export MODEL_ARGS="pretrained=$SAVEPATH,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:$length,temperature:0.6,top_p:0.95}"
export OUTPUT_DIR="/extrahome0/save_dir/GPG/4gpus/${loss_type}/evals"

TASKS=(
    "aime25"
    "math_500"
    "aime24"
)

for TASK in "${TASKS[@]}"; do
    echo "Evaluating task: $TASK"
    log_path=/userhome/Research_HUB/GPG/open-r1/log_dir/${loss_type}/benchmark/${TASK//:/_}/${formatted_time}.log
    echo $log_path
    lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
        --use-chat-template \
        --generation.n=8 \                                   # 每个问题生成 8 个答案
        --evaluation.pass_at_k=[1]   --output-dir "$OUTPUT_DIR/${TASK//:/_}"
done