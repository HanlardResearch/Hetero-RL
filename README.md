An heterogeneous RL algorithm built on [GPG](https://github.com/AMAP-ML/GPG)/[trl](https://github.com/huggingface/trl)/[openR1](https://github.com/huggingface/open-r1).

Asynchronous Reinforcement Learning
```shell
# Enter the current directory (if the directory is different, you need to replace the corresponding path variables in the script).
cd /userhome/Research_HUB/GPG/open-r1

# Launch the learner firstly（using 4 * 80GB Nvidia A100 by default）
CUDA_VISIBLE_DEVICES=2,3,4,5 bash sh_dir/MoIS_Learner_4gpus_nRMs_LogNorm_benchmark.sh learner_script_EqQ_v0_benchmark EqQ_1th 1 v6b EqQ Async_EqQ_diff_32

# Then launch the sampler（using 4 * 80GB Nvidia A100 for each sampler by default）

## Option 1: launch all samplers at once
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sh_dir/MoIS_Learner_4gpus_nRMs_LogNorm_benchmark_checkpoint.sh learner_script_checkpoint GEPO_nothink_1th 1 v6b gepo 1L2S_GEPO_diff32_nothink

## Option 2: launch samplers one by one in sequence
## (Optional) Resume from checkpoint
# please put the path of checkpoint into model_name_or_path
bash sh_dir/MoIS_Sampler_4gpus_single_benchmark_checkpoint.sh sampler_script_checkpoint GEPO_nothink_1th v6b gepo 1L2S_GEPO_diff32_nothink 0 &
bash sh_dir/MoIS_Sampler_4gpus_single_benchmark_checkpoint.sh sampler_script_checkpoint GEPO_nothink_1th v6b gepo 1L2S_GEPO_diff32_nothink 1 &
```


Online-policy（using 4 * 80GB Nvidia A100 by default）:
```shell
# Enter the current directory (if the directory is different, you need to replace the corresponding path variables in the script).
cd /userhome/Research_HUB/GPG/open-r1
# We support grpo/bnpo/dr_grpo/EqP/EqQ/gspo loss currently.
CUDA_VISIBLE_DEVICES="0,1,2,3" MASTER_PORT=29510 bash sh_dir/train_grpo_4gpus_benchmark.sh grpo
```
