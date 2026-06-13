# LLM Experiments

Setup for a local Python virtual environment so dependencies stay isolated from your system Python.

## 1. Create the virtual environment

From the project root:

```bash
python -m venv llm_experiments
```

## 2. Activate the environment

**PowerShell** (project root)

```powershell
.\llm_experiments\Scripts\Activate.ps1
```

**Command Prompt** (project root)

```bat
llm_experiments\Scripts\activate.bat
```

## 3. Install packages

**Base dependencies** (from `requirements.txt`):

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**PyTorch with CUDA 12.6** (optional; use if you want GPU builds from the official wheel index):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If you need a different CUDA version, see [PyTorch — Get Started](https://pytorch.org/get-started/locally/).

## 4. Select the interpreter in Cursor

1. `Ctrl+Shift+P` → **Python: Select Interpreter**
2. Pick `llm_experiments\Scripts\python.exe`

## 5. Jupyter kernel (optional)

Register the venv as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=llm_experiments --display-name "Python (llm_experiments)"
```

In a notebook, choose the **Python (llm_experiments)** kernel.
