model_name=$1

model_dir=trained_models/$model_name

python experiments/seat_prefix.py --save_path $model_dir
python experiments/crows_prefix.py --save_path $model_dir
python experiments/stereoset_prefix.py --save_path $model_dir
