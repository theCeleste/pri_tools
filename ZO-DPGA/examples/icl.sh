MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}

python run.py --model_name $MODEL --task_name $TASK --output_dir result/tmp --tag icl --num_train 32 --num_eval 1000 --load_bfloat16 --verbose "$@"
