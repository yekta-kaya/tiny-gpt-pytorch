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
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale     = math.sqrt(d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(mask, -1e9)

        attn_weights = self.attn_drop(torch.softmax(scores, dim=-1))
        out = self.out_proj(torch.matmul(attn_weights, V))
        return out, attn_weights


class DecoderBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.attn      = MaskedSelfAttention(d_model, dropout=dropout)
        self.ln1       = nn.LayerNorm(d_model)
        self.ffn       = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.ln2       = nn.LayerNorm(d_model)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out, attn_weights = self.attn(x, mask)
        x = self.ln1(x + self.resid_drop(attn_out))
        x = self.ln2(x + self.resid_drop(self.ffn(x)))
        return x, attn_weights


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, hidden_dim=128, num_layers=2, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.emb_drop  = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            DecoderBlock(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)
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