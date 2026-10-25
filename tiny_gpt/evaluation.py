import math

import torch
import torch.nn as nn

from .training import iterate_minibatches


def encode_prompt(prompt_text, stoi):
    prompt_text = prompt_text.lower()
    prompt_text = prompt_text.replace("\n", " ")
    prompt_text = prompt_text.replace(".", " .")
    prompt_text = prompt_text.replace(",", " ,")
    return [stoi[tok] for tok in prompt_text.split() if tok in stoi]


def generate_text(
    model,
    prompt_text,
    stoi,
    itos,
    seq_len,
    device,
    max_new_tokens=10,
    temperature=0.3,
):
    if temperature < 0:
        raise ValueError("Temperature must be non-negative.")

    model.eval()
    prompt_ids = encode_prompt(prompt_text, stoi)

    if not prompt_ids:
        raise ValueError("Prompt has no known tokens in the vocabulary.")

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(x[:, -seq_len:])
            next_token_logits = logits[:, -1, :]

            if temperature == 0:
                # Greedy decoding
                next_id = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                # Temperature sampling
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, next_id], dim=1)

    return " ".join(itos[i] for i in x[0].tolist())


def uniform_random_loss(vocab_size):
    """
    The 'knows nothing' baseline.

    A clueless model gives every token the same probability: p = 1 / vocab_size.
    The cross-entropy loss when the true token had probability p is -log(p).
    Plug in p = 1/V:
        loss = -log(1/V) = log(V)

    We use math.log (natural log) because PyTorch CrossEntropyLoss is in nats.
    """
    return math.log(vocab_size)


def evaluate_lm(model, val_input_ids, val_target_ids, vocab, train_target_ids=None,
                batch_size=32, device=None):
    """
    Compute and explain the standard language-model interpretation metrics.

    The whole idea: cross-entropy loss is in 'nats', which is hard to feel.
    We convert it into more intuitive quantities and compare against baselines
    so you know whether a given loss is good or bad.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    vocab_size = len(vocab)
    criterion = nn.CrossEntropyLoss()

    # ── 1. measure the model's actual loss on validation ─────────────
    loss_sum = correct = total = seen = 0
    with torch.no_grad():
        for xb, yb in iterate_minibatches(val_input_ids, val_target_ids, batch_size, shuffle=False):
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(-1) == yb).sum().item()
            total    += yb.numel()
            seen     += xb.size(0)

    val_loss = loss_sum / max(1, seen)
    val_acc  = correct / max(1, total)

    # ── 2. the two baselines ─────────────────────────────────────────
    # uniform: a model that knows NOTHING, guesses every word equally
    uniform_loss = math.log(vocab_size)

    # unigram: a model that knows word FREQUENCIES but ignores context.
    # Fit on train, scored on val exactly like the real model. Add-one smoothing
    # keeps tokens that never appear in train from giving log(0).
    unigram_loss = None
    if train_target_ids is not None:
        counts    = torch.bincount(train_target_ids.reshape(-1).cpu(), minlength=vocab_size).float() + 1
        log_probs = (counts / counts.sum()).log()
        unigram_loss = -log_probs[val_target_ids.reshape(-1).cpu()].mean().item()

    # ── 3. convert everything to perplexity ──────────────────────────
    # perplexity = exp(loss). It answers: "on average, how many words is the
    # model effectively choosing between?" Lower = more confident & correct.
    val_ppl     = math.exp(val_loss)
    uniform_ppl = math.exp(uniform_loss)
    unigram_ppl = math.exp(unigram_loss) if unigram_loss is not None else None

    # ── 4. how far the model beat random, as a percentage ────────────
    # 0% = no better than random, 100% = perfect (loss 0)
    pct_of_gap_closed = (uniform_loss - val_loss) / uniform_loss * 100

    # ── 5. print a readable report ───────────────────────────────────
    print("=" * 60)
    print("LANGUAGE MODEL EVALUATION")
    print("=" * 60)
    print(f"\nVocab size:              {vocab_size:,}")
    print(f"\n── Your model ──")
    print(f"Validation loss:         {val_loss:.4f} nats")
    print(f"Validation perplexity:   {val_ppl:.1f}")
    print(f"Token accuracy:          {val_acc*100:.2f}%  (next word exactly right)")

    print(f"\n── Baselines (what you're beating) ──")
    print(f"Uniform random loss:     {uniform_loss:.4f}  (perplexity {uniform_ppl:,.0f})")
    print(f"  → knows nothing, picks any of {vocab_size:,} words equally")
    if unigram_loss is not None:
        print(f"Unigram loss:            {unigram_loss:.4f}  (perplexity {unigram_ppl:,.0f})")
        print(f"  → knows word frequencies, but ignores all context")

    print(f"\n── Interpretation ──")
    print(f"Your model effectively chooses between ~{val_ppl:.0f} words per step,")
    print(f"instead of {uniform_ppl:,.0f} (random) — it closed {pct_of_gap_closed:.1f}% of the")
    print(f"gap from random toward perfect prediction.")

    if unigram_loss is not None:
        if val_loss < unigram_loss:
            beat = (unigram_loss - val_loss)
            print(f"\n✓ It beats the unigram baseline by {beat:.2f} nats —")
            print(f"  meaning it genuinely USES CONTEXT, not just word frequency.")
        else:
            print(f"\n✗ It does NOT beat the unigram baseline — the model isn't")
            print(f"  using context yet. It's basically just predicting common words.")
    print("=" * 60)

    return {
        "val_loss":     val_loss,
        "val_ppl":      val_ppl,
        "val_acc":      val_acc,
        "uniform_loss": uniform_loss,
        "uniform_ppl":  uniform_ppl,
        "unigram_loss": unigram_loss,
        "unigram_ppl":  unigram_ppl,
        "pct_gap_closed": pct_of_gap_closed,
    }
