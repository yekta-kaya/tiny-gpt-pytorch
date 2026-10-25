# tiny-gpt

A tiny decoder-only transformer (GPT-style) language model implemented from scratch in PyTorch, as a learning exercise in how attention-based LLMs work end to end.

## Structure

```
tiny_gpt/            reusable library code
  model.py              causal self-attention, positional encoding, TinyGPT decoder stack (no nn.Transformer)
  training.py           training loop: AdamW, LR warmup, gradient clipping, early stopping, checkpointing
  evaluation.py         perplexity/accuracy vs. uniform & unigram baselines, text generation

scripts/
  prepare_data.py       pulls text from Wikipedia + public-domain Project Gutenberg books, tokenizes it

notebooks/
  gpt_experiments.ipynb  experimentation notebook: gather data, train, generate text, evaluate

tests/
  test_model.py          shape/weight-tying/causal-masking checks for TinyGPT

data/                  local corpora (gitignored)
checkpoints/           saved model weights (gitignored)
```

## Results

Trained on ~600K word-level tokens from 23 Wikipedia articles and 7 public-domain Project Gutenberg books (dogs, wolves and other animals). The text is cut into non-overlapping 64-token windows and split 90/10 at random into train/validation.

**Model:** 4 decoder blocks, 4 attention heads, `d_model=128`, FFN 512, context 64 tokens, vocabulary 11,966 words, ~2.3M parameters (embedding and output layer share weights).

| Model | Val loss (nats) | Perplexity | Next-word accuracy |
|---|---:|---:|---:|
| Uniform random (knows nothing) | 9.39 | 11,966 | — |
| Unigram (word frequencies only) | 6.38 | 590 | — |
| **TinyGPT** | **5.05** | **155.6** | **19.87%** |

- On average the model narrows the next word down to ~156 candidates, out of a vocabulary of 11,966.
- It beats the unigram baseline by **1.33 nats**, a **3.8× lower perplexity**. A unigram model only knows how common each word is, so beating it shows the model is actually using context.
- The top-1 prediction is exactly right for about 1 word in 5.

![Training curves](assets/training_curves.png)

**Limitations:** this is a learning project, not a usable LM. The corpus is tiny and narrow, the tokenizer is a simple word-level split (rare words become `<unk>`), and with this little data the model overfits easily, so early stopping keeps the checkpoint with the best validation loss.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and package management.

## 1. Install uv

Follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) if you don't already have it.

## 2. Create the environment and install dependencies

From the project root:

```bash
uv sync
```

This creates a `.venv` and installs the base dependencies pinned in `pyproject.toml`.

## 3. (Optional) Activate the environment

`uv run <command>` works without activating anything, but if you'd rather activate the environment directly:

**PowerShell** (project root)

```powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt** (project root)

```bat
.venv\Scripts\activate.bat
```

## 4. Install PyTorch

PyTorch isn't pinned in `pyproject.toml` since the right wheel depends on your hardware.

CPU-only:

```bash
uv pip install torch
```

Or, for GPU builds from the official wheel index:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If you need a different CUDA version, see [PyTorch — Get Started](https://pytorch.org/get-started/locally/).

## 5. Select the interpreter in your editor

In most editors with a command palette (e.g. `Ctrl+Shift+P`), look for **Python: Select Interpreter** and pick `.venv\Scripts\python.exe`.

## 6. Jupyter kernel (optional)

Register the venv as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name=tiny-gpt --display-name "Python (tiny-gpt)"
```

Open `notebooks/gpt_experiments.ipynb` and choose the **Python (tiny-gpt)** kernel.

## Running tests

```bash
uv run pytest
```

## License

[MIT](LICENSE)
