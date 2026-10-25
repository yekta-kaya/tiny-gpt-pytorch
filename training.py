import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from collections import Counter
from pathlib import Path


def iterate_minibatches(x, y, batch_size, shuffle=True):
    indices = torch.randperm(x.size(0), device=x.device) if shuffle else torch.arange(x.size(0), device=x.device)
    for start in range(0, x.size(0), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def train(
    model,
    train_input_ids,
    train_target_ids,
    val_input_ids,
    val_target_ids,
    num_epochs=10,
    batch_size=32,
    lr=3e-4,                 # standard GPT starting point
    weight_decay=0.01,       # regularization (AdamW)
    warmup_frac=0.1,         # fraction of total steps spent warming up
    grad_clip=1.0,           # gradient clipping max norm
    log_every=500,           # print a progress line every N batches
    patience=2,              # stop if val loss doesn't improve for this many epochs
    ckpt_path="checkpoints/best.pt",
    device=None,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)   # ensure dir exists

    vocab_size = model.lm_head.out_features
    criterion  = nn.CrossEntropyLoss()

    # AdamW — decoupled weight decay, the transformer standard
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),   # 0.95 beta2 is the GPT convention
    )

    steps_per_epoch = max(1, math.ceil(train_input_ids.size(0) / batch_size))
    total_steps  = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps   # ramp up
        return 1.0                       # flat after warmup

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    lrs = []

    best_val_loss = float("inf")
    epochs_without_improve = 0

    for epoch in range(num_epochs):
        # ── train ────────────────────────────────────────────────
        model.train()
        loss_sum = correct = total = seen = 0
        win_loss = win_correct = win_total = win_seen = 0   # window accumulators

        print(f"\nEpoch {epoch+1:02d}/{num_epochs}")
        print("-" * 65)

        for batch_idx, (xb, yb) in enumerate(
            iterate_minibatches(train_input_ids, train_target_ids, batch_size, shuffle=True), 1
        ):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()

            # epoch accumulators
            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(-1) == yb).sum().item()
            total    += yb.numel()
            seen     += xb.size(0)

            # window accumulators
            win_loss    += loss.item() * xb.size(0)
            win_correct += (logits.argmax(-1) == yb).sum().item()
            win_total   += yb.numel()
            win_seen    += xb.size(0)

            if batch_idx % log_every == 0:
                avg_loss = win_loss / max(1, win_seen)        # exact, handles partial batches
                avg_acc  = win_correct / max(1, win_total)
                current_lr = lr_scheduler.get_last_lr()[0]
                print(
                    f"  batch {batch_idx:>6}/{steps_per_epoch} | "
                    f"loss {avg_loss:.4f}  acc {avg_acc:.4f} | "
                    f"lr {current_lr:.2e}"
                )
                win_loss = win_correct = win_total = win_seen = 0   # reset window

        train_losses.append(loss_sum / max(1, seen))
        train_accs.append(correct / max(1, total))
        lrs.append(lr_scheduler.get_last_lr()[0])

        # validation
        model.eval()
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

        val_losses.append(loss_sum / max(1, seen))
        val_accs.append(correct / max(1, total))

        print("-" * 65)
        print(
            f"  epoch summary | "
            f"train loss {train_losses[-1]:.4f}  acc {train_accs[-1]:.4f} | "
            f"val loss {val_losses[-1]:.4f}  acc {val_accs[-1]:.4f} | "
            f"lr {lrs[-1]:.2e}"
        )

        # ── checkpoint + early stopping ──────────────────────────
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_without_improve = 0
            torch.save({
                "epoch":        epoch + 1,
                "model_state":  model.state_dict(),
                "optim_state":  optimizer.state_dict(),
                "val_loss":     best_val_loss,
                "train_losses": train_losses,
                "val_losses":   val_losses,
            }, ckpt_path)
            print(f"best model saved (val loss {best_val_loss:.4f}) → {ckpt_path}")
        else:
            epochs_without_improve += 1
            print(f"  no improvement ({epochs_without_improve}/{patience})")
            if epochs_without_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} — "
                      f"best val loss {best_val_loss:.4f}")
                break

    # ── plots ─────────────────────────────────────────────────────
    actual_epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(actual_epochs, train_losses, marker="o", label="Train")
    axes[0].plot(actual_epochs, val_losses,   marker="o", label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(actual_epochs, train_accs, marker="o", label="Train")
    axes[1].plot(actual_epochs, val_accs,   marker="o", label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Token Accuracy"); axes[1].legend()

    axes[2].plot(actual_epochs, lrs, marker="o", color="green")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR"); axes[2].set_title("Learning Rate")

    plt.suptitle("Train vs. Validation")
    plt.tight_layout()
    plt.show()

    return train_losses, val_losses, train_accs, val_accs


def encode_prompt(prompt_text, stoi):
    prompt_text = prompt_text.lower()
    prompt_text = prompt_text.replace("\n", " ")
    prompt_text = prompt_text.replace(".", " .")
    prompt_text = prompt_text.replace(",", " ,")
    return [stoi[tok] for tok in prompt_text.split() if tok in stoi]


def generate_text(model, prompt_text, stoi, itos, seq_len, device, max_new_tokens=30):
    model.eval()
    prompt_ids = encode_prompt(prompt_text, stoi)
    if not prompt_ids:
        raise ValueError("Prompt has no known tokens in the vocabulary.")
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(x[:, -seq_len:])
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
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

    # unigram: a model that knows word FREQUENCIES but ignores context
    unigram_loss = None
    if train_target_ids is not None:
        counts = torch.bincount(train_target_ids.reshape(-1), minlength=vocab_size).float()
        probs  = counts / counts.sum()
        probs  = probs[probs > 0]
        unigram_loss = -(probs * probs.log()).sum().item()

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