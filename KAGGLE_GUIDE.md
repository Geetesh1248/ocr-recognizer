# Kaggle Training Guide for Text Recognizer (Phase 3)

This guide provides complete, step-by-step instructions to train the upgraded `TextRecognizerV2` (5.57M parameters, ResNet + 2-layer BiLSTM + CTC) on a free Kaggle GPU (NVIDIA Tesla T4).

---

## 1. Overview & Curriculum

We train the text recognizer from random initialisation (zero pretrained weights, zero external OCR APIs) using a 3-stage curriculum:

| Stage | Data Source | Sample Size / Epoch | Target Split for Early Stopping | Epochs & LR | Stopping Rule |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage A** | 100% Synthetic (Streaming generator or bundled fonts) | 50,000 samples / epoch (or 20k from manifest) | Synthetic Val (unseen font families) | 30 epochs, `lr = 1e-3` | Stops if Synthetic Val CER does not improve for 5 epochs. Saves `recognizer_stageA_best.pt`. |
| **Stage B** | Synthetic mixed with ALL Real Train crops | **75,430 samples** (ALL 37,715 real FUNSD train crops + 37,715 synthetic) | Real FUNSD Val CER (strictly real crops) | 25 epochs, `lr = 5e-4` | Stops if Real Val CER does not improve for 5 epochs. Saves `recognizer_stageB_best.pt`. Requires `stageA_best.pt`. |
| **Stage C** | 100% Real FUNSD Train crops only | **37,715 samples** (real train crops only, 0 synthetic) | Real FUNSD Val CER (strictly real crops) | 15 epochs, `lr = 1e-4` | Stops if Real Val CER does not improve for 5 epochs. Saves `recognizer_stageC_best.pt`. Requires `stageB_best.pt`. |

---

## 2. Preparing the Upload Bundle

We provide an automated bundle generator that packages all necessary files:
- Source code (`increment_*`, `train_recognizer_gpu.py`, `evaluate_recognizer.py`, `common.py`, etc.)
- Configuration files (`config.json`, `config_words.json`, `config_lines.json`)
- All 56 open-source font files (`fonts/`) and licenses (`fonts/LICENSES.md`)
- Plain-text English corpus (`corpus/`)
- Best trained detector checkpoint (`models/detector_20epochs.pt`)
- Manifest files (`datasets/recognition_*.jsonl`)
- Real FUNSD crops (`datasets/recognition_real/`)

### Create the Bundle Locally:
Run the following PowerShell command in the project root:
```powershell
python make_kaggle_bundle.py
```
This generates `outputs/kaggle_recognizer_bundle.zip` (~161 MB).

---

## 3. Kaggle Setup (Step-by-Step)

### Step 3.1: Create Kaggle Dataset
1. Log into [Kaggle](https://www.kaggle.com).
2. Click **+ Create** -> **New Dataset**.
3. Title your dataset: `ocr-recognizer-bundle`.
4. Upload `outputs/kaggle_recognizer_bundle.zip`.
5. Click **Create**.

### Step 3.2: Create Kaggle Notebook
1. Go to **Code** -> **+ New Notebook**.
2. **Import or Create**: You can click **File -> Import Notebook** and select `kaggle_train.ipynb` (which is bundled in this repository and runs the entire training end-to-end), or create a fresh notebook.
3. On the right-side panel under **Notebook options**:
   - **Accelerator**: Select **GPU T4 x2** or **GPU T4**.
   - **Persistence**: Select **Variables and Files** (optional, recommended).
   - **Internet**: Toggle **On** (needed only for initial `pip install` checks).
3. Under **Input**, click **+ Add Input** -> Search for `ocr-recognizer-bundle` -> Click the `+` icon to add it.

---

## 4. Notebook Execution Commands

### Cell 1: Environment Setup & Unpack
Unpack the dataset into your writable working directory (`/kaggle/working`):

```python
import os
import shutil
import zipfile
from pathlib import Path

# Paths
INPUT_ZIP = "/kaggle/input/ocr-recognizer-bundle/kaggle_recognizer_bundle.zip"
WORKDIR = Path("/kaggle/working/ocr_project")

if not WORKDIR.exists():
    print(f"Unpacking {INPUT_ZIP} to {WORKDIR}...")
    WORKDIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(INPUT_ZIP, "r") as zf:
        zf.extractall(WORKDIR)
    print("Unpack complete!")

os.chdir(WORKDIR)
print("Current Working Directory:", os.getcwd())
!python check_env.py
```

### Cell 2: Sanity Check & Unit Tests
Verify project integrity and run the test suite:

```bash
python check_project.py
python -m unittest tests/test_recognizer_v2.py
```

---

## 5. Running Training Stages

### Stage A: Synthetic Training (Random Initialization)
Trains on streaming synthetic text across 51 training font families:

```bash
python train_recognizer_gpu.py \
    --stage A \
    --epochs 30 \
    --batch-size 64 \
    --lr 1e-3 \
    --synth-count 50000 \
    --workers 2 \
    --amp
```
- **Saves to**: `models/recognizer_stageA_best.pt` and `models/recognizer_stageA_epochXXX.pt`.
- **Logs to**: `outputs/recognizer_train_log.csv` and `outputs/recognizer_train_log.json`.

### Stage B: Curriculum Mix (50/50 Synthetic + Real)
Initializes from Stage A best weights and mixes real FUNSD train crops:

```bash
python train_recognizer_gpu.py \
    --stage B \
    --epochs 25 \
    --batch-size 64 \
    --lr 5e-4 \
    --synth-count 25000 \
    --resume models/recognizer_stageA_best.pt \
    --workers 2 \
    --amp
```
- **Saves to**: `models/recognizer_stageB_best.pt`.

### Stage C: Fine-tuning on Real FUNSD Train Crops
Fine-tunes strictly on real document crops with a low learning rate:

```bash
python train_recognizer_gpu.py \
    --stage C \
    --epochs 15 \
    --batch-size 64 \
    --lr 1e-4 \
    --resume models/recognizer_stageB_best.pt \
    --workers 2 \
    --amp
```
- **Saves to**: `models/recognizer_stageC_best.pt`.

---

## 6. Resuming After Disconnection

Kaggle GPU sessions have a 12-hour limit and may disconnect. The training script supports full state resumption:
- Saves model weights, optimizer state, OneCycle/Cosine scheduler state, GradScaler state, epoch count, and best CER.

To resume an interrupted run (e.g. Stage A at epoch 14):
```bash
python train_recognizer_gpu.py \
    --stage A \
    --epochs 30 \
    --resume models/recognizer_stageA_epoch014.pt \
    --amp
```

---

## 7. Model Evaluation & Language Model Decoding

After Stage C completes, evaluate the model on validation data:

### Standard Evaluation (Greedy CTC)
```bash
python evaluate_recognizer.py \
    --checkpoint models/recognizer_stageC_best.pt \
    --split val \
    --output outputs/val_evaluation_greedy.json
```

### Enhanced Evaluation (CTC Beam Search + Char N-gram LM)
```bash
python evaluate_recognizer.py \
    --checkpoint models/recognizer_stageC_best.pt \
    --split val \
    --beam-width 10 \
    --lm-weight 0.4 \
    --output outputs/val_evaluation_beam.json
```

### Final Held-Out Test Split (Run ONCE Only)
> [!IMPORTANT]
> The test split must only be evaluated once at the very end. The script enforces `--final-test` as a safety guard.

```bash
python evaluate_recognizer.py \
    --checkpoint models/recognizer_stageC_best.pt \
    --split test \
    --final-test \
    --beam-width 10 \
    --lm-weight 0.4 \
    --output outputs/test_evaluation_final.json
```

---

## 8. Downloading Artifacts from Kaggle

To download the trained checkpoints and training logs:

```python
import shutil

# Zip output models and logs
shutil.make_archive("/kaggle/working/recognizer_outputs", "zip", "/kaggle/working/ocr_project/models")
print("Download recognizer_outputs.zip from the Output tab on the right panel.")
```

---

## 9. Throughput & Timing Estimates

| Metric | Measured on Local CPU | Estimated on Kaggle Tesla T4 GPU |
| :--- | :--- | :--- |
| **Synthetic Rendering** | 344.9 samples/sec (single thread) | ~600–800 samples/sec (2 workers) |
| **Forward + Backward Pass** | ~14.7 samples/sec (8 CPU threads) | ~3,500–5,000 samples/sec (CUDA AMP) |
| **Stage A (50k synth samples)** | ~55 minutes / epoch | **~1.5 – 2.0 minutes / epoch** |
| **Stage B (50k mixed samples)** | ~45 minutes / epoch | **~1.2 – 1.8 minutes / epoch** |
| **Stage C (57k real samples)** | ~35 minutes / epoch | **~0.8 – 1.2 minutes / epoch** |
| **Full 3-Stage Training Run** | ~48 hours (prohibitive) | **~1.5 – 2.5 hours total** |

*Note: GPU throughput estimates are based on ResNet-BiLSTM fp16 throughput benchmarks on NVIDIA T4; local CPU numbers are empirically measured.*
