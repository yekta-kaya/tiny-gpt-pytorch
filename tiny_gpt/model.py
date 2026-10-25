import math
import torch
import torch.nn as nn


def make_causal_mask(seq_len, device):
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale     = math.sqrt(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)

    def _split_heads(self, t):
        # (B, L, d_model) → (B, H, L, head_dim)
        B, L, _ = t.shape
        return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, mask):
        B, L, D = x.shape
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # (B, H, L, L)
        scores = scores.masked_fill(mask, -1e9)                      # (L, L) mask broadcasts over B, H

        attn_weights = self.attn_drop(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn_weights, V)                          # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)         # merge heads back
        return self.out_proj(out), attn_weights


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.ln1       = nn.LayerNorm(d_model)
        self.attn      = MaskedSelfAttention(d_model, num_heads, dropout=dropout)
        self.ln2       = nn.LayerNorm(d_model)
        self.ffn       = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        # pre-LN (GPT-2 style): normalize the sublayer input, keep the residual path clean
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + self.resid_drop(attn_out)
        x = x + self.resid_drop(self.ffn(self.ln2(x)))
        return x, attn_weights


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, hidden_dim=128, num_layers=2, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.emb_drop  = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            DecoderBlock(d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)                       # final norm before head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # bias=False for tying

        # weight tying: share the embedding table with the output projection
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # GPT-2 style init: small normal for linears/embeddings, zeros for biases
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, L = input_ids.shape
        mask = make_causal_mask(L, input_ids.device)

        x = self.emb_drop(self.pos_enc(self.token_emb(input_ids)))

        last_attn_weights = None
        for block in self.blocks:
            x, last_attn_weights = block(x, mask)

        x = self.ln_f(x)                                          # apply final norm
        return self.lm_head(x), last_attn_weights