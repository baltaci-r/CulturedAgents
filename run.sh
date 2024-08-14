config_file_or_env="OAI_CONFIG_LIST"

mode=${1}
filter=${2}

save_dir=runs/$mode/$filter
mkdir -p $save_dir

python main.py \
  --save_dir $save_dir \
  --config_file_or_env $config_file_or_env \
  --mode $mode \
  --filter $filter