SQUAD_DIR=$HOME/DRCD


python -m torch.distributed.launch --nproc_per_node=1 ./examples/run_squad.py \
    --model_type bert \
    --model_name_or_path $HOME/models/bert-base-uncased\
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/DRCD_training.json \
    --predict_file $SQUAD_DIR/DRCD_dev.json \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --save_steps 3000 \
    --logging_steps 3000 \
    --output_dir $HOME/models/DRCD_from_en/ \
    --eval_all_checkpoints \
    --per_gpu_eval_batch_size=4   \
    --fp16
