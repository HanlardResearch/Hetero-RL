An heterogeneous RL algorithm built on [GPG](https://github.com/AMAP-ML/GPG)/[trl](https://github.com/huggingface/trl)/[openR1](https://github.com/huggingface/open-r1).

# Asynchronous Reinforcement Learning

## Enter the current directory (if the directory is different, you need to replace the corresponding path variables in the script).


## Launch the learner firstly（using 4 * 80GB Nvidia A100 by default）
```shell
cd /userhome/Research_HUB/GPG/open-r1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sh_dir/Learner_4gpus_nRMs_LogNorm_benchmark_checkpoint.sh learner_script_checkpoint GEPO_nothink_1th 1 v6b gepo 1L2S_GEPO_diff32_nothink
```
## Sampler: launch samplers one by one in sequence
### resume from checkpoint: put the path of checkpoint into model_name_or_path
```shell
bash sh_dir/Sampler_4gpus_single_benchmark_checkpoint.sh sampler_script_checkpoint GEPO_nothink_1th v6b gepo 1L2S_GEPO_diff32_nothink 0 &
bash sh_dir/Sampler_4gpus_single_benchmark_checkpoint.sh sampler_script_checkpoint GEPO_nothink_1th v6b gepo 1L2S_GEPO_diff32_nothink 1 &
```


Online-policy（using 4 * 80GB Nvidia A100 by default）:

# We support grpo/bnpo/dr_grpo/gepo/gspo loss currently.
```shell
cd /userhome/Research_HUB/GPG/open-r1
CUDA_VISIBLE_DEVICES="0,1,2,3" MASTER_PORT=29510 bash sh_dir/Online_gXpo_4gpus_benchmark.sh gepo
```
