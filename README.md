基于[GPG](https://github.com/AMAP-ML/GPG)/[trl](https://github.com/huggingface/trl)/[openR1](https://github.com/huggingface/open-r1) 改进的异构算法

运行脚本
```shell
#进入当前目录（目录不同则需要替换脚本的部分路径变量）
cd /userhome/Research_HUB/GPG/open-r1

# 先启动学习器
bash sh_dir/MoIS_Learner_4gpus_nRMs.sh learner_script_MoIS_v3c MoISv3_7th 1 v0c1 is_bnpo_prompt_clip_enhanced_1L4S &

# 再按启动采样器
bash sh_dir/MoIS_Sampler_4gpus.sh sampler_script_MoIS_v3c MoISv3_7th v0c1 is_bnpo_prompt_clip_enhanced_1L4S
```