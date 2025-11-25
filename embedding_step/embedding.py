import torch

input_ids = torch.tensor([2,3,5,1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size,output_dim)

# input_ids の各 ID を対応する埋め込みベクトルに変換
# 内部的には、embedding_layer.weight という行列から
# 行番号が input_ids に対応するベクトルを取り出しているだけ
print(embedding_layer(input_ids))