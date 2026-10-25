import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


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

        # ── validation ───────────────────────────────────────────
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
