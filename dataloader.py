# for the training set, convert everything into a history, replies
# for the validation set, convert everything into a history, replies
import time
import torch
from datasets import load_dataset
import torch.distributed as dist


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = self._prepare(dataset)

    def _prepare(self, dataset):
        index = []
        for page_i, page in enumerate(self.dataset):
            for pair_i, pair in enumerate(page["qa_pairs"]):
                index.append((page_i, pair_i))
        return index

    def __len__(self):
        return sum([e["num_pairs"] for e in self.dataset])
    
    def __getitem__(self, idx):
        page_i, pair_i = self.index[idx]
        page = self.dataset[page_i]
        pair = page["qa_pairs"][pair_i]
        pair["page_id"] = page["id"]
        return pair


class MonolingualDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, *, batch_size, seed = 42, epoch = 0, single_domain = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.single_domain = single_domain

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset)
        # return sum([e["num_pairs"] for e in self.dataset])

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            seen_domains = set()
            for i in torch.randperm(len(self.dataset), generator=g).tolist():
                page = self.dataset[i]
                if page["domain"] in  seen_domains:
                    continue
                for pair in page["qa_pairs"]:
                    pair["page_id"] = page["id"]
                    yield pair
                if self.single_domain:
                    seen_domains.add(page["domain"])
            self.epoch += 1


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, datasets, languages, *, probabilities = None, batch_size = 10, seed = 42, single_domain = False, alpha = 0.3):
        self.datasets = [MonolingualDataset(e, batch_size=batch_size, single_domain=single_domain) for e in datasets]
        self.languages = languages
        self.probabilities = self._get_default_probs(alpha) if probabilities is None else torch.Tensor(probabilities) 
        self.batch_size = batch_size
        self.seed = seed
        self.alpha = alpha

    def _get_default_probs(self, alpha):
        ds_length = [len(e) for e in self.datasets]
        total_length = sum(ds_length)
        probs = [(e/total_length)**alpha for e in ds_length]
        return torch.Tensor(probs)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        self.datasets = [iter(e) for e in self.datasets]
        while True:
            idx = torch.multinomial(self.probabilities, 1)[0].item()
            for _ in range(self.batch_size):
                item = next(self.datasets[idx])
                yield item


