import urllib.request
from gpt_download import download_and_load_gpt2
from gpt_model.gpt_model import GPTModel
from pre_training.config import GPT_CONFIG_124M, model_configs
from pre_training.load_weights_into_gpt import load_weights_into_gpt
import torch
from pre_training.util import generate, text_to_token_ids, token_ids_to_text
import tiktoken

url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/"
       "01_main-chapter-code/gpt_download.py")
filename = url.split("/")[-1]
urllib.request.urlretrieve(url, filename)

settings,params = download_and_load_gpt2(model_size="124M",models_dir="gpt2")

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()

load_weights_into_gpt(gpt,params)
gpt.to("cpu")

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you",tokenizer).to("cpu"),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:\n",token_ids_to_text(token_ids,tokenizer))