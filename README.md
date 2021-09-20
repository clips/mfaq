# MFAQ: a Multilingual FAQ Dataset

MFAQ is a multilingual FAQ retrieval dataset. We also release a multilingual FAQ retrieval model trained on this dataset.

## Dataset
The dataset is hosted on the HuggingFace hub. You can find it [here](https://huggingface.co/datasets/clips/mfaq).

Start by installing the dataset package:
```
pip install datasets
```

Then import the dataset:
```python
from datasets import load_dataset
en_dataset = load_dataset("clips/mfaq", "en")
```
You can find more information about the dataset and the available configurations on the [description page](https://huggingface.co/datasets/clips/mfaq).

## Model
The pre-trained FAQ retrieval model is also hosted on the HuggingFace Hub. You can find it [here](https://huggingface.co/clips/mfaq).

Start by installing sentence-transformers:
```
pip install sentence-transformers
```

Load the model:
```python
from sentence_transformers import SentenceTransformer
```

Each question must be pre-pended with a `<Q>`, answers with a `<A>`.
```python
question = "<Q>How many models can I host on HuggingFace?"
answers = [
  "<A>All plans come with unlimited private models and datasets.",
  "<A>AutoNLP is an automatic way to train and deploy state-of-the-art NLP models, seamlessly integrated with the Hugging Face ecosystem.",
  "<A>Based on how much training data and model variants are created, we send you a compute cost and payment link - as low as $10 per job."
]
model = SentenceTransformer('clips/mfaq')
q_embedding, *a_embeddings = model.encode([question] + answers)
best_answer_idx = sorted(enumerate(a_embeddings), key=lambda x: q_embedding.dot(x[1]), reverse=True)[0][0]
print(answers[best_answer_idx])
```

## Training
`train.py` uses the [HuggingFace Trainer](https://huggingface.co/transformers/main_classes/trainer.html) to train a FAQ retrieval model on MFAQ.
The following configuration reaches an MRR of 89% on the English subset:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py \
    --languages en de es fr it nl pt tr ru pl id no sv da vi fi ro cs he hu hr  \
    --output_dir output/1024 \
    --do_train \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --distributed_softmax \
    --max_steps 3000 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --max_seq_len 128 \
    --warmup_steps 1000 \
    --label_names page_id \
    --logging_steps 5 \
    --fp16 \
    --metric_for_best_model eval_global_mrr \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --report_to tensorboard \
    --dataloader_num_workers 1 \
    --single_domain \
    --hidden_dropout_prob 0.25 \
    --learning_rate 0.00005 \
    --weight_decay 0.01 \
    --alpha 1 \
    --gradient_checkpointing \
    --deepspeed config/ds_config_zero_2.json
```
