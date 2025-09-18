#!/bin/bash

# 检查是否传入两个参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <start> <end>"
    echo "Example: $0 0 7"
    exit 1
fi

start=$1
end=$2

# 检查是否为整数
if ! [[ "$start" =~ ^-?[0-9]+$ ]] || ! [[ "$end" =~ ^-?[0-9]+$ ]]; then
    echo "Error: Both arguments must be integers."
    exit 1
fi

# 循环从 start 到 end（包含）
for ((i=start; i<=end; i++)); do
    device_id=$((i % 8))
    master_port=$((29524 + i))

    echo "==> Running with ASCEND_RT_VISIBLE_DEVICES=$device_id, MASTER_PORT=$master_port, last arg: $i"
    ASCEND_RT_VISIBLE_DEVICES=$device_id \
    MASTER_PORT=$master_port \
    bash sh_dir/Sampler_4gpus_single_benchmark_checkpoint_fixing_obs_npu.sh \
        sampler_script_checkpoint_fixing_obs_npu \
        GEPO_nothink_9999th_debug_npu \
        v6b \
        gepo \
        1L2S_GEPO_diff32_nothink_debug_9999_npu \
        $i
    if [ $i -lt $end ]; then
        echo "==> Waiting 60 seconds before next job... ($(date '+%Y-%m-%d %H:%M:%S'))"
        sleep 60
    fi
done

echo "All jobs completed."