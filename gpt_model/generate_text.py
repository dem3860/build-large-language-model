import torch 
import tiktoken
from gpt_model.gpt_model import GPTModel
from gpt_model.config import GPT_CONFIG_124M

def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

            #最後のステップのみに注目
            logits = logits[:,-1,:]

            probas = torch.softmax(logits,dim=-1)

            idx_next = torch.argmax(probas,dim=-1,keepdim=True)

            idx = torch.cat((idx,idx_next),dim=1)
    
    return idx

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("Encoded start context:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

model = GPTModel(GPT_CONFIG_124M)

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:",out)
print("Output length:", len(out[0]))

decoded = tokenizer.decode(out[0].tolist())
print("Decoded output:", decoded)
