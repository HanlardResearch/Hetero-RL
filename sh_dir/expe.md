### cppo beta ablation
## 1.1
```bash
# launch learner
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sh_dir/Learner_4gpus_GEPO_Ablation.sh learner_script_checkpoint gepo_abl_cppobeta_0_001 1 exp gepo HyperGroup5-GEPO-diff32-1.7B-2k-cppobeta-0.001 8 0.001

# launch sampler 0 on GPU 4/5/6/7
bash sh_dir/Sampler_4gpus_GEPO_Ablation.sh sampler_script_checkpoint gepo_abl_cppobeta_0_001 exp gepo HyperGroup5-GEPO-diff32-1.7B-2k-cppobeta-0.001 8 0.001 0 &

# launch sampler 1 on GPU 4/5/6/7
bash sh_dir/Sampler_4gpus_GEPO_Ablation.sh sampler_script_checkpoint gepo_abl_cppobeta_0_001 exp gepo HyperGroup5-GEPO-diff32-1.7B-2k-cppobeta-0.001 8 0.001 1 &
```

## 1.2
```bash

# launch learner
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sh_dir/Learner_4gpus_GEPO_Ablation.sh learner_script_checkpoint gepo_abl_cppobeta_0_01 1 exp gepo HyperGroup5-GEPO-diff32-1.7B-2k-cppobeta-0.01 8 0.01

# launch sampler 0 on GPU 4/5/6/7
bash sh_dir/Sampler_4gpus_GEPO_Ablation.sh sampler_script_checkpoint gepo_abl_cppobeta_0_01 exp gepo HyperGroup5-GEPO-diff32-1.7B-2k-cppobeta-0.01 8 0.01 0 &

# launch sampler 1 on GPU 4/5/6/7
bash sh_dir/Sampler_4gpus_GEPO_Ablation.sh sampler_script_checkpoint gepo_abl_cppobeta_0_01 exp gepo HyperGroup5-GEPO-diff32-1.7B-2k-cppobeta-0.01 8 0.01 1 &
```

