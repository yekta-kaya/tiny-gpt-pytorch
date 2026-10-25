import torch

from tiny_gpt import TinyGPT


def test_forward_output_shape():
    vocab_size, seq_len, batch = 37, 8, 4
    model = TinyGPT(vocab_size=vocab_size, d_model=16, hidden_dim=32, num_layers=2, max_len=seq_len)
    x = torch.randint(0, vocab_size, (batch, seq_len))

    logits, attn_weights = model(x)

    assert logits.shape == (batch, seq_len, vocab_size)
    assert attn_weights.shape == (batch, seq_len, seq_len)


def test_weight_tying():
    model = TinyGPT(vocab_size=20, d_model=8, hidden_dim=16, num_layers=1, max_len=8)
    assert model.lm_head.weight is model.token_emb.weight


def test_causal_mask_blocks_future_tokens():
    torch.manual_seed(0)
    vocab_size, seq_len = 10, 5
    model = TinyGPT(vocab_size=vocab_size, d_model=8, hidden_dim=16, num_layers=1, max_len=seq_len)
    model.eval()

    x = torch.randint(0, vocab_size, (1, seq_len))
    with torch.no_grad():
        _, attn_weights = model(x)

    # position i must not attend to any position j > i
    upper = torch.triu(attn_weights[0], diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper))
