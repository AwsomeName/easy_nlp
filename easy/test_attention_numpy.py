import numpy as np

def softmax(x):
    x_ep = np.exp(x - np.max(x))
    out = x_ep / np.sum(x_ep, axis=-1, keepdims=True)
    return out


def attention(x, head=12):
    n, dim = x.shape
    wq = np.random.rand(dim, dim)
    wk = np.random.rand(dim, dim)
    wv = np.random.rand(dim, dim)

    vq = x @ wq
    vk = x @ wk
    vv = x @ wv
    
    A = vq @ vk.T
    A = A / np.sqrt(dim)
    A = softmax(A)
    out = A @ vv
    return out
    
def multi_head_attention(x, head=12):
    n, dim = x.shape
    assert dim % head == 0
    head_dim = dim / head
    wq = np.random.rand(dim, dim)
    wk = np.random.rand(dim, dim)
    wv = np.random.rand(dim, dim)

    q = x @ wq
    k = x @ wk
    v = x @ wv

    q = np.reshape(q, (n, head, head_dim))
    k = np.reshape(k, (n, head, head_dim))
    v = np.reshape(v, (n, head, head_dim))
    
    q = np.transpose(q, (1,0,2)) // head, n, head_dim
    k = np.transpose(k, (1,0,2)) // head, n, head_dim
    v = np.transpose(v, (1,0,2)) // head, n, head_dim
    
    
    A = q @ np.transpose(k, [0, 2, 1]) // head, n, head_dim * head, head_dim, n
    A = A / np.sqrt(head_dim) // head, n, n
    A = softmax(A)
    out = A @ v // head,n, head_dim
    out = np.transpose(out, (1,0,2))
    out = np.reshape(out, (n, dim))
    return out
    


# if __name__ == "__main__":
#     # x = np.random()
#     attention(np.random.rand(512,768))



import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, max_seq_len, n_embed, n_heads):
        super().__init__()
        assert n_embed % n_heads == 0
        
        self.head_dim = n_embed // n_heads
        self.max_seq_len = max_seq_len
        self.n_embed = n_embed
        self.n_heads = n_heads
        
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
        )

        self.q_mtx = nn.Linear(n_embed, n_embed, bias=False)
        self.k_mtx = nn.Linear(n_embed, n_embed, bias=False)
        self.v_mtx = nn.Linear(n_embed, n_embed, bias=False)
        self.o_mtx = nn.Linear(n_embed, n_embed, bias=False)


    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        print(x)
        q = self.q_mtx(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_mtx(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_mtx(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1 / math.sqrt(self.head_dim)
        attn_logits = (q @ k.transpose(-1, -2)) * scale
        

        mask = self.mask[:seq_len, :seq_len]
        attn_logits = attn_logits.masked_fill(mask==0, float('-inf'))


        attn = F.softmax(attn_logits, dim=-1)
        
        out = attn @ v
        out = out.transpose(1,2).contiguous()
        out = out.view(batch_size, seq_len, self.n_embed)

        out = self.o_mtx(out)
        return out


cac = CausalSelfAttention(128, 256, 2)
x = np.random.rand(4, 128, 256)
cac.forward(torch.tensor(x, dtype=torch.float32))



