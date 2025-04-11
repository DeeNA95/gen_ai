#! /usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import string

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f'device available: {device}')

#env
batch_size = 256
block_size = 256
max_iters = 500
eval_iters = 100
n_embd = 256 # embedding dimension
lr = 1e-4
HEAD_SIZE = 32
N_HEADS = 4


with open('GPT/data/wikitext-2-raw/wiki.train.raw', 'r') as f:
    text = f.read()

with open('GPT/data/wikitext-2-raw/wiki.valid.raw', 'r') as f:
    valid = f.read()

print(f'First 1000 characters \n {text[:1000]}\n\n')

#getting characters for vocab

chars = sorted(list(set(text+valid)))

def filter_english_chars(vocab_string):
    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + ' ' + '\n') # Include space
    filtered_vocab = ''.join(c for c in vocab_string if c in allowed_chars)
    return filtered_vocab, allowed_chars

vocab, chars = filter_english_chars(''.join(chars))

print(f'characters available in vocab {vocab}')

def filter_text_by_vocab(text_var, allowed_chars):
    filtered_text = ''.join(c for c in text_var if c in allowed_chars)
    return filtered_text

text = filter_text_by_vocab(text, chars)
valid = filter_text_by_vocab(valid, chars)

#tokenisation
#simple char level tokenisation
s_to_i={ch: i for i, ch in enumerate(chars)}
i_to_s={i: ch for i, ch in enumerate(chars)}

encode = lambda s: [s_to_i[c] for c in s]
decode = lambda l: ''.join([i_to_s[i] for i in l])

vocab_size = len(chars)

# # using tiktoken
# import tiktoken
# encoding = tiktoken.get_encoding("gpt2")
# print(encoding.encode("hello world"))

# print(encoding.encode("goodbye world"))
# print(encoding.decode(encoding.encode("hello world")))
# vocab_size = encoding.n_vocab

# tokenise the data

data = torch.tensor(encode(text), dtype=torch.long, device=device)
test = torch.tensor(encode(valid), dtype=torch.long, device=device)

# dataloading

def get_batch(split):
    batch_data = data if split == 'train' else test
    ix = torch.randint(len(batch_data) - block_size, (batch_size,))
    x = torch.stack([batch_data[i:i+block_size] for i in ix])
    y = torch.stack([batch_data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class Head(nn.Module):
    """single head of self attention"""
    def __init__(self, HEAD_SIZE):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = (q @ k.transpose(-2, -1)) * (C ** -0.5) # B x T x T
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B x T x T
        weights = F.softmax(weights, dim=-1)

        output = weights @ v # B x T x C
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1) #concatenated over the channel dim

# Baseline Bigram Model

class BiGram(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.linear_layer = nn.Linear(n_embd, vocab_size)
        self.sa_head = MultiHeadAttention(HEAD_SIZE/N_HEADS, N_HEADS)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        token_embeddings = self.token_embedding_table(idx) # B x T x n_embd tensor
        postion_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # T x n_embd tensor
        x = token_embeddings + postion_embeddings # B x T x n_embd tensor

        x = self.sa_head(x) # apply self attention head to get output

        # logits is a batch_size x block_size x vocab_size tensor
        logits = self.linear_layer(x) # apply linear layer to get logits


        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C) # reshape to match pytorchs crossentropy loss function
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # calculate the loss

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # generate new tokens
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # last time step logits  since bigram  B,C
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) #B,1
            idx = torch.cat((idx, idx_next), dim=1) # B, T+1
        return idx

#init model

model = BiGram()
model.to(device)

# init optimiser
optimizer = optim.AdamW(model.parameters(), lr=lr)

# loss esitmator

@torch.no_grad()
def estimate_loss():

    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# training loop
for iter in range(max_iters):

    if iter % 100 == 0:
        losses = estimate_loss()
        print(f"iter {iter}: train loss {losses['train']:.4f}, valid loss {losses['val']:.4f}")

    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training complete")

print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))
