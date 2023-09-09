# adapter from https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py
import os
import sys
import torch
import logging
import pandas as pd
import torch.distributed as dist
from typing import Optional, List
from dataclasses import dataclass, field
from datasets import load_dataset, interleave_datasets
from transformers import set_seed, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments, HfArgumentParser
from transformers import AutoModel, AutoTokenizer
from collections import OrderedDict
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from accelerate.utils import DistributedType


from dataloader import IterableDataset, ValidationDataset


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    #gradient_checkpointing: bool = field(default=False)
    hidden_dropout_prob: float = field(default=0.1)
    attention_probs_dropout_prob: float = field(default=0.1)
    model_name_or_path: str = field(default="xlm-roberta-base")
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")


@dataclass
class DataTrainingArguments:
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_seq_length: int = field(default=None)
    languages: Optional[List[str]] = field(default=None)
    probabilities: Optional[List[float]] = field(default=None)
    overwrite_cache: bool = field(default=False)
    pad_to_max_length: bool = field(default=False)
    single_domain: bool = field(default=False)
    alpha: float = field(default=0.3)
    no_special_token: bool = field(default=False)
    limit_valid_size: Optional[int] = field(default=1000)
    limit_train_size: Optional[int] = field(default=1000)

@dataclass
class CustomTrainingArgument(TrainingArguments):
    distributed_softmax: bool = field(default=False)


def distributed_softmax(q_output, a_output, rank, world_size):
    q_list = [torch.zeros_like(q_output) for _ in range(world_size)]
    a_list = [torch.zeros_like(a_output) for _ in range(world_size)]
    dist.all_gather(tensor_list=q_list, tensor=q_output.contiguous())
    dist.all_gather(tensor_list=a_list, tensor=a_output.contiguous())
    q_list[rank] = q_output
    a_list[rank] = a_output
    q_output = torch.cat(q_list, 0)
    a_output = torch.cat(a_list, 0)
    return q_output, a_output


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output["last_hidden_state"] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        page_id = inputs.pop("page_id", None)
        outputs = model(**inputs)
        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        q_logits, a_logits = torch.chunk(sentence_embeddings, 2)
        if self.args.distributed_softmax and self.args.local_rank != -1 and return_outputs is False:
            q_logits, a_logits = distributed_softmax(
                q_logits, a_logits, self.args.local_rank, self.args.world_size
            )
            labels = torch.arange(q_logits.size(0), device=a_logits.device)
        cross_entropy = torch.nn.CrossEntropyLoss()
        dp = q_logits.mm(a_logits.transpose(0, 1))
        labels = torch.arange(dp.size(0), device=dp.device)
        loss = cross_entropy(dp, labels)
        if return_outputs:
            outputs = OrderedDict({"q_logits": q_logits, "a_logits": a_logits, "page_id": page_id})
        return (loss, outputs) if return_outputs else loss


def get_acc_rr(q_logits, a_logits):
    q_logits = torch.from_numpy(q_logits)
    a_logits = torch.from_numpy(a_logits)
    dp = q_logits.mm(a_logits.transpose(0, 1))
    indices = torch.argsort(dp, dim=-1, descending=True)
    targets = torch.arange(indices.size(0), device=indices.device).view(-1, 1)
    targets = targets.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    acc = ranks.eq(1).float().squeeze()
    rr = torch.reciprocal(ranks).squeeze()
    return rr, acc


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArgument))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_seed(training_args.seed)

    model_kwargs = dict(
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
        attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
        add_pooling_layer=False
    )

    #if model_args.gradient_checkpointing:
        # CANINE does not supporte
    #    model_kwargs["gradient_checkpointing"] = True

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        additional_special_tokens=None if data_args.no_special_token else ["<Q>", "<A>", "<link>"]
    )
    if not data_args.no_special_token:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    datasets = [load_dataset("clips/mfaq", l) for l in data_args.languages]
    
    """if data_args.limit_train_size:
        train_datasets = [e["train"].select(range(data_args.limit_valid_size)) for e in datasets]  # Limit the training data to 1000 rows
    else:
        train_datasets = [e["train"] for e in datasets]

    if data_args.limit_valid_size:
        eval_datasets = [e["validation"].select(range(data_args.limit_valid_size)) for e in datasets]
    else:
        eval_datasets = [e["validation"] for e in datasets]"""

    train_datasets = [e["train"] for e in datasets]
    eval_datasets = [e["validation"] for e in datasets]

    if data_args.limit_train_size:
        train_datasets = [e.select(range(data_args.limit_train_size)) for e in train_datasets]  # Limit the training data to 1000 rows
    if data_args.limit_valid_size:
        eval_datasets = [e.select(range(data_args.limit_valid_size)) for e in eval_datasets]

    eval_dataset = ValidationDataset(interleave_datasets(eval_datasets))

    if training_args.do_train:
        world_size = 1 if training_args.world_size is None else training_args.world_size
        train_dataset = IterableDataset(
            train_datasets,
            data_args.languages, 
            probabilities=data_args.probabilities, 
            batch_size=training_args.per_device_train_batch_size*world_size,
            seed=training_args.seed,
            single_domain=data_args.single_domain,
            alpha=data_args.alpha
        )

    padding = "max_length" if data_args.pad_to_max_length else True
    def collate_fn(batch):
        questions, answers, page_ids = [], [], []
        for item in batch:
            questions.append(item['question'] if data_args.no_special_token else f"<Q>{item['question']}")
            answers.append(item['answer'] if data_args.no_special_token else f"<A>{item['answer']}")
            page_ids.append(item["page_id"])
        output = tokenizer(
            questions + answers,
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            return_tensors="pt",
            pad_to_multiple_of=8
        )
        output["page_id"] = torch.Tensor(page_ids)
        return output

    def compute_metrics(predictions):
        q_output, a_output, page_id = predictions.predictions
        unique_page_ids = set(page_id.tolist())
        global_rr, global_acc, pp_mrr, pp_acc = [], [], [], []
        for unique_page_id in unique_page_ids:
            selector = page_id == unique_page_id
            s_q_output = q_output[selector, :]
            s_a_output = a_output[selector, :]
            rr, acc = get_acc_rr(s_q_output, s_a_output)
            global_rr.append(rr)
            global_acc.append(acc)
            pp_mrr.append(rr.mean())
            pp_acc.append(acc.mean())
        global_mrr = torch.cat(global_rr).mean()
        global_acc = torch.cat(global_acc).mean()
        per_page_mrr = torch.stack(pp_mrr).mean()
        per_page_acc = torch.stack(pp_acc).mean()
        return {"global_mrr": global_mrr, "global_acc": global_acc, "per_page_mrr": per_page_mrr, "per_page_acc": per_page_acc}

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    if training_args.do_predict:
        logger.info("*** Predict ***")
        _, _, metrics = trainer.predict(eval_dataset, metric_key_prefix="predict")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()