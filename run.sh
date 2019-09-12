SQUAD_DIR=$HOME/sentence_squad/DRCD


python -m torch.distributed.launch --nproc_per_node=1 ./examples/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-chinese \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/DRCD_training.json \
    --predict_file $SQUAD_DIR/DRCD_dev.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --save_steps 1000 \
    --output_dir ../models/DRCD/ \
    --overwrite_output_dir \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --fp16
