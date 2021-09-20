#!/bin/sh
for language in en de es fr it nl pt tr ru pl id no sv da vi fi ro cs he hu hr
do
    echo "Evaluation for ${language}"
    output_dir="output/eval/${language}" 
    python train.py \
        --languages $language \
        --model_name_or_path $MODEL \
        --output_dir $output_dir \
        --do_predict \
        --per_device_eval_batch_size $BATCH_SIZE \
        --max_seq_len 128 \
        --label_names page_id \
        --dataloader_num_workers 1
done
