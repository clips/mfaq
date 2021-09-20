#!/bin/sh
for language in de es fr it pt tr ru pl
do
    echo "Evaluation for ${language}"
    output_dir="output/eval/mono/${language}"
    model="output/mono/${language}/checkpoint-1500"
    python train.py \
        --languages $language \
        --model_name_or_path $model \
        --output_dir $output_dir \
        --do_prPedict \
        --per_device_eval_batch_size $BATCH_SIZE \
        --max_seq_len 128 \
        --label_names page_id \
        --dataloader_num_workers 1
done
