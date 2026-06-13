import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def iterate_minibatches(x, y, batch_size, shuffle=True):
    indices = torch.randperm(x.size(0), device=x.device) if shuffle else torch.arange(x.size(0), device=x.device)
    for start in range(0, x.size(0), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


import math

def train(
    model,
    train_input_ids,
    train_target_ids,
    val_input_ids,
    val_target_ids,
    num_epochs=10,
    batch_size=32,
    lr=3e-4,                 # lowered from 1e-3 — standard GPT starting point
    weight_decay=0.01,       # regularization
    warmup_frac=0.1,         # fraction of total steps spent warming up
    grad_clip=1.0,           # gradient clipping max norm
    device=None,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    vocab_size = model.lm_head.out_features
    criterion  = nn.CrossEntropyLoss()

    # AdamW instead of Adam — decoupled weight decay, strictly better for transformers
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),   # 0.95 beta2 is the GPT convention
    )

    # ---- LR scheduler: linear warmup -> cosine decay (per-step) ----
    steps_per_epoch = max(1, math.ceil(train_input_ids.size(0) / batch_size))
    total_steps  = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        # cosine decay from 1.0 down to ~0 over remaining steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    lrs = []

    for epoch in range(num_epochs):
        model.train()
        loss_sum = correct = total = seen = 0

        for xb, yb in iterate_minibatches(train_input_ids, train_target_ids, batch_size, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
            loss.backward()

            # gradient clipping — standard for transformer stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()          # step the LR every batch

            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(-1) == yb).sum().item()
            total    += yb.numel()
            seen     += xb.size(0)

        train_losses.append(loss_sum / max(1, seen))
        train_accs.append(correct / max(1, total))
        lrs.append(scheduler.get_last_lr()[0])

        # ---- validation ----
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
                seen     += xb.size(0)          # accumulate ACTUAL val samples

        val_losses.append(loss_sum / max(1, seen))   # fixed: divide by val samples seen
        val_accs.append(correct / max(1, total))

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Train Loss: {train_losses[-1]:.4f}  Acc: {train_accs[-1]:.4f} | "
            f"Val Loss: {val_losses[-1]:.4f}  Acc: {val_accs[-1]:.4f} | "
            f"LR: {lrs[-1]:.2e}"
        )

    # ---- plots ----
    epochs = range(1, num_epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(epochs, train_losses, marker="o", label="Train")
    axes[0].plot(epochs, val_losses,   marker="o", label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(epochs, train_accs, marker="o", label="Train")
    axes[1].plot(epochs, val_accs,   marker="o", label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Token Accuracy"); axes[1].legend()

    axes[2].plot(epochs, lrs, marker="o", color="green")
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