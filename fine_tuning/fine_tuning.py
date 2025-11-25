import tiktoken
from fine_tuning.dataset import SpamDataset
from torch.utils.data import DataLoader
import torch
from gpt_download import download_and_load_gpt2
from gpt_model.gpt_model import GPTModel
from pre_training.load_weights_into_gpt import load_weights_into_gpt 
from pre_training.util import generate_text_simple, text_to_token_ids, token_ids_to_text
from fine_tuning.util import calc_accuracy_loader,calc_loss_batch,calc_loss_loader

tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(
    csv_file="fine_tuning/data/train.csv",
    tokenizer=tokenizer,
    max_length=None,
)

val_dataset = SpamDataset(
    csv_file="fine_tuning/data/validation.csv",
    tokenizer=tokenizer,
    max_length=train_dataset.max_length,
)
test_dataset = SpamDataset(
    csv_file="fine_tuning/data/test.csv",
    tokenizer=tokenizer,
    max_length=train_dataset.max_length,
)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)


# ----------- config　設定 -----------
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# ------- モデルを読み込む -------
model_size = CHOOSE_MODEL.split(" ")[-1].strip("(").rstrip(")")
settings,params = download_and_load_gpt2(
    model_size=model_size,models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model,params)
model.eval()

for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
num_classes = 2
# 出力層をファインチューニング用に分類クラス数に置き換え
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes,
)

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    train_loader,model,device,num_batches=10
)

val_accuracy = calc_accuracy_loader(
    val_loader,model,device,num_batches=10
)   

test_accuracy = calc_accuracy_loader(
    test_loader,model,device,num_batches=10
)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader,model,device,num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader,model,device,num_batches=5
    )
    test_loss = calc_loss_loader(
        test_loader,model,device,num_batches=5
    )

print(f"Training loss: {train_loss:.4f}")
print(f"Validation loss: {val_loss:.4f}")
print(f"Test loss: {test_loss:.4f}")