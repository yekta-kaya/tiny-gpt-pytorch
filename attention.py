import math
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from bs4 import BeautifulSoup


def make_causal_mask(seq_len, device):
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, L, D)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x, mask):
        # x: (B, L, D)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # (B, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # mask: (L, L), True means block
        scores = scores.masked_fill(mask, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = self.out_proj(out)

        return out, attn_weights


class DecoderBlock(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.attn = MaskedSelfAttention(d_model)
        self.ln1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_out, attn_weights = self.attn(x, mask)
        x = self.ln1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)

        return x, attn_weights


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, hidden_dim=128, num_layers=2, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model=d_model, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # input_ids: (B, L)
        B, L = input_ids.shape
        mask = make_causal_mask(L, input_ids.device)

        x = self.token_emb(input_ids)   # (B, L, D)
        x = self.pos_enc(x)             # (B, L, D)

        last_attn_weights = None
        for block in self.blocks:
            x, last_attn_weights = block(x, mask)

        logits = self.lm_head(x)        # (B, L, vocab_size)
        return logits, last_attn_weights


def iterate_minibatches(x, y, batch_size, shuffle=True):
    indices = torch.randperm(x.size(0), device=x.device) if shuffle else torch.arange(x.size(0), device=x.device)

    for start in range(0, x.size(0), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]




def fetch_wikipedia_text(urls, verbose=True):
    """
    Fetch and concatenate text from multiple Wikipedia URLs.

    Args:
        urls (list[str]): list of Wikipedia links
        verbose (bool): print progress

    Returns:
        str: concatenated cleaned text
    """

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    all_texts = []

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()

            if verbose:
                print(f"\nFetched: {url}")
                print(f"Status: {response.status_code}")

            soup = BeautifulSoup(response.text, "html.parser")

            # More reliable: focus on main content
            content = soup.find("div", {"id": "mw-content-text"})

            if content is not None:
                paragraphs = content.find_all("p")
            else:
                paragraphs = soup.find_all("p")

            paragraphs = [p.get_text(strip=True) for p in paragraphs]
            paragraphs = [p for p in paragraphs if p]

            if verbose:
                print(f"Paragraphs: {len(paragraphs)}")

            page_text = " ".join(paragraphs)

            # basic cleanup
            page_text = page_text.lower()
            page_text = page_text.replace("\n", " ")
            page_text = " ".join(page_text.split())

            all_texts.append(page_text)

        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

    final_text = " ".join(all_texts)

    return final_text

