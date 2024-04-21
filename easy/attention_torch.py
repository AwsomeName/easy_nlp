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
            # `torch.tril`是方法，作用在已经有的tensor上
            "mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
        )
        
        self.q_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.k_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.v_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.o_proj = nn.Linear(n_embed, n_embed, bias=False)
    
    
    def forward(self, x):
        # x.shape: [batch_size, seq_len, n_embed]
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 注意：开根号的是head_dim
        # 注意：一般scale写成乘法，所以需要计算 1 / sqrt(head_dim)
        scale = 1 / math.sqrt(self.head_dim)
        
        # 交换了才能乘，去掉中间的head_dim维度(for each head)
        # attn_logits.shape: [batch_size, self.n_heads, seq_len, seq_len]
        attn_logits = (q @ k.transpose(-1, -2)) * scale
        
        # 产生casual mask, 作用在最后一维上
        mask = self.mask[:seq_len, :seq_len]
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        
        # attn.shape: [batch_size, self.n_heads, seq_len, seq_len]
        #   `F.softmax`最好加上dim-1，不然会抛警告
        attn = F.softmax(attn_logits, dim=-1) 
        
        # attn.shape: [batch_size, self.n_heads, seq_len, seq_len]
        # v.shape: [batch_size, self.n_heads, seq_len, self.head_dim]
        # -> 
        # 去掉中间的seq_len维度(for each position)
        # out.shape: [batch_size, self.n_heads, seq_len, self.head_dim]
        out = attn @ v
        
        # 把head维度换到后面去，拼接
        # out.shape: [batch_size, seq_len, self.n_heads, self.head_dim]
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.n_embed)
        
        # 拼接完做个等宽映射
        out = self.o_proj(out)
        return out