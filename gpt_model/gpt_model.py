import torch 
import torch.nn as nn
import tiktoken
from gpt_model.config import GPT_CONFIG_124M
from gpt_model.transformer import TransformerBlock
from gpt_model.layer_norm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最後に適用される層正規化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 線形出力層
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"],bias=False)
    
    def forward(self,in_idx):
        batch_size, seq_len = in_idx.shape
        # print("in_idx.shape:", in_idx.shape)
        tok_embeds = self.tok_emb(in_idx)
        # print("tok_embeds.shape:", tok_embeds.shape)
        pos_embeds = self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        # print("pos_embeds.shape:", pos_embeds.shape)
        x  = tok_embeds + pos_embeds
        # print("x after sum shape:", x.shape)
        x = self.drop_emb(x)
        # print("x after dropout shape:", x.shape)
        x = self.trf_blocks(x)
        # print("x after transformer blocks shape:", x.shape)
        x = self.final_norm(x)
        # print("x after final norm shape:", x.shape)
        logits = self.out_head(x)
        return logits
    
if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch,dim=0)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("output shape:", logits.shape)  # (2, seq_len, vocab_size)
    print("logits:", logits)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in GPTModel: {total_params:,}")
