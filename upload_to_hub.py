from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers import SentenceTransformer


parser = ArgumentParser()
parser.add_argument("--checkpoint")
parser.add_argument("--model_name", default="mfaq")
parser.add_argument("--organization", default="clips")
parser.add_argument("--exist_ok", action="store_true")
parser.add_argument("--replace_model_card", action="store_true")
args = parser.parse_args()


# model = AutoModel.from_pretrained(args.checkpoint, add_pooling_layer=False)
# tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

model = Transformer(
    args.checkpoint, 
    max_seq_length=128, 
    model_args={"add_pooling_layer": False},
    tokenizer_name_or_path=args.checkpoint
)
pooling = Pooling(model.auto_model.config.hidden_size, pooling_mode="mean")
st = SentenceTransformer(modules=[model, pooling])
st.save_to_hub(
    args.model_name, 
    organization=args.organization,
    exist_ok=args.exist_ok,
    replace_model_card=args.replace_model_card
)

