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
