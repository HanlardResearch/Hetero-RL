An heterogeneous RL algorithm built on [GPG](https://github.com/AMAP-ML/GPG)/[trl](https://github.com/huggingface/trl)/[openR1](https://github.com/huggingface/open-r1).

# Asynchronous Reinforcement Learning

## Enter the current directory (if the directory is different, you need to replace the corresponding path variables in the script).

## Launch the learner firstly（using 4 * 80GB Nvidia A100 by default）
```shell
cd ./open-r1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sh_dir/HeteroRL_Learner_4gpus.sh learner_script_checkpoint GEPO_think_1th 1 v6b gepo 1L2S_GEPO_diff32_think
```
## Sampler: launch samplers one by one in sequence
### resume from checkpoint: put the path of checkpoint into model_name_or_path
```shell
bash sh_dir/HeteroRL_Sampler_4gpus.sh sampler_script_checkpoint GEPO_think_1th v6b gepo 1L2S_GEPO_diff32_think 0 &
bash sh_dir/HeteroRL_Sampler_4gpus.sh sampler_script_checkpoint GEPO_think_1th v6b gepo 1L2S_GEPO_diff32_think 1 &
bash sh_dir/HeteroRL_Sampler_4gpus.sh sampler_script_checkpoint GEPO_think_1th v6b gepo 1L2S_GEPO_diff32_think 2 &
bash sh_dir/HeteroRL_Sampler_4gpus.sh sampler_script_checkpoint GEPO_think_1th v6b gepo 1L2S_GEPO_diff32_think 3 &
```


Online-policy（using 4 * 80GB Nvidia A100 by default）:

# We support grpo/bnpo/dr_grpo/gepo/gspo loss currently.
```shell
cd /userhome/Research_HUB/GPG/open-r1
CUDA_VISIBLE_DEVICES="0,1,2,3" MASTER_PORT=29510 bash sh_dir/Online_gXpo_4gpus.sh gepo
```
