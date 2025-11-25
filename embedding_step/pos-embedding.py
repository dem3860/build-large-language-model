import torch
from torch.utils.data import Dataset,DataLoader
import tiktoken

torch.manual_seed(123)

class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

with open("embedding-step/the-verdict.txt", "r", encoding="utf-8") as f:
   raw_text = f.read()

vocab_size = 50257
output_dim = 256

# 語彙サイズ分の学習可能な埋め込みベクトルを持つ層
token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:", inputs)

# 8 * 4 * 256
# 1バッチで4トークン、それを8バッチ繰り返す。1トークンあたりの埋め込みベクトルの次元は256
token_embeddings = token_embedding_layer(inputs)

# 絶対位置埋め込み
context_length = max_length
# 4 * 256
pos_embedding_layer = torch.nn.Embedding(context_length,output_dim)
# 同一バッチ内のtokenに対して位置がわかるようになる。pos_embeddings[0]はそれぞれのbatchにおける最初のトークンの位置埋め込み
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

