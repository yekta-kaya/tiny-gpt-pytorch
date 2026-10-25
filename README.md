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

data/                  local corpora (gitignored — see below)
checkpoints/           saved model weights (gitignored)
```

## Data

`data/` isn't committed — it's gitignored, same as `checkpoints/`. Generate your own corpus by running:

```bash
uv run python scripts/prepare_data.py
```

This scrapes the default set of Wikipedia pages and Gutenberg books and writes `data/dog_corpus.txt`. Edit the `_DEFAULT_WIKIPEDIA_URLS` / `_DEFAULT_GUTENBERG_IDS` lists in `scripts/prepare_data.py`, or call `gather_data(wikipedia_urls=..., gutenberg_ids=...)` yourself, to use a different corpus.

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
