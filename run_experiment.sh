model_dir=$1

cuda=1

filelist=`ls $model_dir | grep epoch`


for f in $filelist
do 
    echo $model_dir"/"$f

    CUDA_VISIBLE_DEVICES=$cuda  python experiments/crows_prefix.py --save_path $model_dir"/"$f --split test

    #CUDA_VISIBLE_DEVICES=$cuda  python experiments/crows_prefix.py --save_path $model_dir"/"$f --split clean_test

    CUDA_VISIBLE_DEVICES=$cuda  python experiments/perplexity_prefix.py --save_path $model_dir"/"$f

    #CUDA_VISIBLE_DEVICES=$cuda  python experiments/seat_prefix.py --save_path $model_dir"/"$f

    CUDA_VISIBLE_DEVICES=$cuda  python experiments/stereoset_prefix.py --save_path $model_dir"/"$f --split dev
    CUDA_VISIBLE_DEVICES=$cuda  python experiments/stereoset_prefix.py --save_path $model_dir"/"$f --split test
    #CUDA_VISIBLE_DEVICES=$cuda  python experiments/stereoset_prefix.py --save_path $model_dir"/"$f --split clean_test

    CUDA_VISIBLE_DEVICES=$cuda  python experiments/regard_prefix.py --save_path $model_dir"/"$f
    
done


