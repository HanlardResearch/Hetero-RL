sleep 16200
ps -ef | grep learner  | grep -v grep | awk '{print $2}' | xargs kill -9
sleep 1
ps -ef | grep sampler | grep -v grep | awk '{print $2}' | xargs kill -9
sleep 1

cd /userhome/Research_HUB/GPG/open-r1

CUDA_VISIBLE_DEVICES=0,1,2,3 bash sh_dir/Learner_4gpus_nRMs_LogNorm_benchmark_checkpoint.sh learner_script_checkpoint GEPO_nothink_2th 1 v6b gepo 1L2S_GEPO_diff32_nothink_2th

bash sh_dir/Sampler_4gpus_single_benchmark_checkpoint.sh sampler_script_checkpoint GEPO_nothink_2th v6b gepo 1L2S_GEPO_diff32_nothink_2th 0 &

sleep 60

bash sh_dir/Sampler_4gpus_single_benchmark_checkpoint.sh sampler_script_checkpoint GEPO_nothink_2th v6b gepo 1L2S_GEPO_diff32_nothink_2th 1 &
