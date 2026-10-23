# Environment Setup & Workflow

## Requirements

- Python 3.10 – 3.12 x64
- Allowed dependencies: PyTorch, NumPy, Pillow, OpenCV, pypdfium2, pyclipper
- **Hard Rule**: No pretrained OCR weights, no ready-made OCR engines (Tesseract, EasyOCR, PaddleOCR, etc.). All models train from random initialisation.

---

## 1. Setup on Windows PowerShell (Primary)

From this directory (`FULLY SELF MADE`):

```powershell
# Create a fresh virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run the environment check
python check_env.py

# Verify code integrity and run unit tests
python check_project.py
python -m unittest discover
```

If using an existing virtual environment located in the parent directory:
```powershell
..\.venv\Scripts\python.exe check_env.py
..\.venv\Scripts\python.exe check_project.py
..\.venv\Scripts\python.exe -m unittest discover
```

---

## 2. Setup on Linux / macOS (Secondary)

From the project root directory:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify environment, integrity, and tests
python check_env.py
python check_project.py
python -m unittest discover
```

---

## Device & Hardware Acceleration

The system automatically detects whether an NVIDIA GPU with CUDA is available:
- If `torch.cuda.is_available()` is `True`, training runs on `cuda` and prints `Using device: cuda`.
- Otherwise, it falls back to `cpu` and prints `Using device: cpu`.
- You can override device selection with `--device cpu` or `--device cuda`.

---

## Running the Pipeline

### First image preparation (Increment 1)

```powershell
python cli.py run examples\demo_page.pdf --until 1 --output outputs\prepared_01
```

Linux / macOS equivalent:
```bash
python cli.py run examples/demo_page.pdf --until 1 --output outputs/prepared_01
```

Select pages with `--pages 1,3`, or supply a known clockwise rotation with `--rotation 90`. Use a new or empty output folder for every run.

### Later Increments (2 through 5)

Increment 2 uses our trained word detector by default:
- Default config: `config.json` points to `models/detector_20epochs.pt` (`annotation_level: "word"`).
- Explicit word mode: `python cli.py --config config_words.json run examples\demo_page.pdf --until 2 --output outputs\words_demo`
- Line mode (requires line model training): `python cli.py --config config_lines.json run examples\demo_page.pdf --until 2 --output outputs\lines_demo`
- Full pipeline `--until 5` requires training the text recognizer (`models/recognizer.pt`) first. Missing checkpoints produce explicit errors rather than fake outputs.
