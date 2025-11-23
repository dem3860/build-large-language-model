import os
import requests
import tiktoken

if not os.path.exists("embedding/the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    file_path = "embedding/the-verdict.txt"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)
    
with open("embedding/the-verdict.txt", "r", encoding="utf-8") as f:
   raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

enc_text = tokenizer.encode(raw_text)
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
print(f"x:{x}")
print(f"y:     {y}")

for i in range(1,context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"Context: {tokenizer.decode(context)} -> Next token: {tokenizer.decode([desired])}")