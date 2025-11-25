import os
import requests
import re

# ローカルになければダウンロード
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
    
# print("Total number of character:", len(raw_text))
# print(raw_text[:99])

# トークン化
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

vocab = {token:integer for integer, token in enumerate(all_words)}

class SimpleTokenizerV1:
    def __init__(self,vocab):
        # 語彙 : トークン -> 整数
        self.str_to_int = vocab
        # 逆語彙 : 整数 -> トークン
        self.int_to_str = {i : s for s,i in vocab.items()}
    
    def encode(self,text):
        # トークン化
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # 前後の空白を除去し、空文字を除去
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # トークンを整数に変換
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self,ids):
        text = ' '.join([self.int_to_str[i] for i in ids])
        print(f"Decoded text: {text[:50]}...")
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# idsをもとにデコード
print(tokenizer.decode(ids))