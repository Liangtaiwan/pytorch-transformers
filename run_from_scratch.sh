SQUAD_DIR=$HOME/DRCD


python ./examples/run_squad.py \
    --model_type bert \
    --config_name bert-base-chinese \
    --tokenizer_name bert-base-chinese \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/DRCD_training.json \
    --predict_file $SQUAD_DIR/DRCD_dev.json \
    --learning_rate 3e-5 \
    --num_train_epochs 50 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps 3000 \
    --save_steps 3000 \
    --evaluate_during_training \
    --output_dir $HOME/models/DRCD_from_scratch/ \
    --overwrite_output_dir \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --fp16
