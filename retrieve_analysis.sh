formatted_time=$(date "+%Y%m%d-%H-%M-%S")
log_path=/userhome/Research_HUB/GPG/open-r1/log_dir/retrieve_and_analysis/${formatted_time}.log
sync_weights_path=$1
max_num_model_weight=64
num_samples=64
num_generations=8
echo $log_path
# --skip_retrieve_model_weight
nohup python retrieve_and_analysis.py --sync_weights_path $sync_weights_path \
    --num_samples $num_samples --num_generations $num_generations \
    --max_num_model_weight $max_num_model_weight --random_dataset > $log_path 2>&1 &

