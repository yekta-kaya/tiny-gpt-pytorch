# tiny-gpt

A tiny decoder-only transformer (GPT-style) language model implemented from scratch in PyTorch, as a learning exercise in how attention-based LLMs work end to end.

- **`attention.py`** — causal self-attention, positional encoding, and a minimal `TinyGPT` decoder stack built from first principles (no `nn.Transformer`).
- **`data_gathering.py`** — pulls training text from Wikipedia articles and public-domain Project Gutenberg books, then tokenizes/vectorizes it into training sequences.
- **`training.py`** — training loop with AdamW, LR warmup, gradient clipping, early stopping, checkpointing, and perplexity/accuracy evaluation against uniform and unigram baselines.
- **`attention_trial.ipynb`** — notebook tying it together: gather data, train, generate text, evaluate.

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

## 4. Install PyTorch with CUDA 12.6 (optional)

Use this if you want GPU builds from the official wheel index:

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

In a notebook, choose the **Python (tiny-gpt)** kernel.
