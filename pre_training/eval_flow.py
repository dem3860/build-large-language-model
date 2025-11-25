import torch
import tiktoken
from gpt_model.gpt_model import GPTModel
from pre_training.config import GPT_CONFIG_124M
from pre_training.util import token_ids_to_text

if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")

    inputs = torch.tensor([[16833,3626,6100], # "Every effort moves"
                           [40,1107,588]]) # "I really like"
    
    targets = torch.tensor([[3626,6100,345], # "effort moves you"
                            [1107,588,11311]]) # "really like chocolate"
    with torch.no_grad():
        logits = model(inputs)

    # ---------------- 予測結果の表示 ----------------
    probas = torch.softmax(logits,dim=-1)
    
    token_ids = torch.argmax(probas,dim=-1,keepdim=True)
    print("Token IDs:\n", token_ids)

    print(f"Targets batch 1: {token_ids_to_text(targets[0],tokenizer)}")
    print(f"Outputs batch 1: "f"{token_ids_to_text(token_ids[0].flatten(),tokenizer)}")

    print(f"Targets batch 2: {token_ids_to_text(targets[1],tokenizer)}")
    print(f"Outputs batch 2: "f"{token_ids_to_text(token_ids[1].flatten(),tokenizer)}")

    text_idx = 0
    target_probas_1 = probas[text_idx,[0,1,2],targets[text_idx]]
    print("Target token probabilities batch 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx,[0,1,2],targets[text_idx]]
    print("Target token probabilities batch 2:", target_probas_2)

# ---------------- cross-entropy ----------------
    # バッチ次元で結合してフラット化しておく (logitsとtargetsの次元を一致させてcross-entropy計算用に)
    logits_flat = logits.flatten(0,1)
    targets_flat = targets.flatten()

    loss = torch.nn.functional.cross_entropy(logits_flat,targets_flat)
    print("Cross-entropy loss:", loss.item())

    


