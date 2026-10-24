# Validation record

## Phase 3 Completion — Three-Stage Recognizer Curriculum & Held-Out Test Evaluation

All 57 tests pass (`python -m unittest discover -s tests`) and the full source/JSON audit passes (`python check_project.py`).
The recognizer (`TextRecognizerV2`, 5,565,981 parameters, ResNet + 2-layer BiLSTM + CTC loss) was trained entirely from random weights through the 3-stage curriculum on a Kaggle Tesla T4 GPU with zero pretrained weights or external OCR engines.

### 1. Three-Stage Curriculum Training Summary

| Curriculum Stage | Training Data Composition | Target Epochs | Completed Epoch | Initial LR | Final Train Loss | Best Val CER | Early Stopping Criterion |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage A** | 100% on-the-fly streaming synthetic crops (50k/epoch, training fonts only) | 30 | 19 | $1.00\times 10^{-3}$ | 0.11614 | **0.02987** | Patience=5 on Synthetic Unseen-Font Val CER |
| **Stage B** | 50% streaming synthetic (37,715/ep) + 50% real FUNSD train crops (37,715) | 25 | 19 | $5.00\times 10^{-4}$ | 0.11614 | **0.04132** | Patience=5 on Real FUNSD Val CER |
| **Stage C** | 100% real FUNSD train crops (37,715 ground truth words & lines) | 15 | 7 | $1.00\times 10^{-4}$ | 0.00976 | **0.04135** | Patience=5 on Real FUNSD Val CER |

### 2. Validation Split Evaluation (6,346 Samples)

Evaluated on `datasets/recognition_val.jsonl` (2,000 synthetic crops with unseen validation fonts + 4,346 real FUNSD crops):

| Evaluation Subset | Sample Count | Stage B Best (`recognizer_stageB_best.pt`) | Stage C Best / Default (`models/recognizer.pt`) |
| :--- | :---: | :---: | :---: |
| **Synthetic Val (Unseen Fonts)** | 2,000 | CER: **0.0227** \| WER: **0.0753** | CER: **0.0238** \| WER: **0.0817** |
| **Real FUNSD Val (Combined)** | 4,346 | CER: **0.0413** \| WER: **0.1728** | CER: **0.0413** \| WER: **0.1697** |
| **Real FUNSD GT Word Crops** | 3,202 | CER: **0.0365** \| WER: **0.1230** | CER: **0.0370** \| WER: **0.1249** |
| **Real FUNSD GT Line Crops** | 1,144 | CER: **0.0456** \| WER: **0.2227** | CER: **0.0452** \| WER: **0.2146** |

**Decoder Tuning (Greedy CTC vs. Beam Search + Character 4-gram LM)**:
- Baseline Greedy CTC: CER = **0.0362**, WER = **0.1570** (on real validation test slice).
- Beam Search (`width=5`, `lm_weight=0.20`): CER = **0.0367**, WER = **0.1597**.
- Beam Search (`width=5`, `lm_weight=0.40`): CER = **0.0365**, WER = **0.1570**.
- **Conclusion**: Greedy CTC is faster and optimal on structured form fields, while Beam Search provides language smoothing. Stage C was copied to `models/recognizer.pt` as the default production model due to superior document-level line recognition (WER 21.46% vs 22.27%) and real overall WER (16.97% vs 17.28%).

### 3. Final Held-Out Real FUNSD Test Split Evaluation (11,336 Samples)

Executed **once** on `datasets/recognition_test.jsonl` with `--final-test` using `models/recognizer.pt` (`outputs/test_metrics_final.json`):

| Test Set Breakdown | Sample Count | Total Characters | Total Words | Final Greedy CER | Final Greedy WER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Real FUNSD Test (Combined)** | **11,336** | **65,856** | **11,885** | **0.0736 (7.36%)** | **0.2633 (26.33%)** |
| **Real FUNSD GT Word Crops** | **8,651** | **37,707** | **8,651** | **0.0709 (7.09%)** | **0.2171 (21.71%)** |
| **Real FUNSD GT Line Crops** | **2,685** | **28,149** | **3,234** | **0.0761 (7.61%)** | **0.3095 (30.95%)** |

**Held-Out Test CER by Character Length Buckets**:
- Length 1–5: 5,967 samples | CER = **0.0829**
- Length 6–10: 3,511 samples | CER = **0.0701**
- Length 11–20: 1,203 samples | CER = **0.0650**
- Length 21+: 655 samples | CER = **0.0754**

**Top 10 Character Confusions on Held-Out Test Set**:
1. `' ' -> <deleted/sub>`: 165 times
2. `'.' -> <deleted/sub>`: 145 times
3. `'o' -> <deleted/sub>`: 139 times
4. `'i' -> <deleted/sub>`: 134 times
5. `'t' -> <deleted/sub>`: 123 times
6. `'n' -> <deleted/sub>`: 121 times
7. `'e' -> <deleted/sub>`: 120 times
8. `'a' -> <deleted/sub>`: 109 times
9. `'r' -> <deleted/sub>`: 96 times
10. `'I' -> <deleted/sub>`: 95 times
Seven new unit tests in `tests/test_recognizer_v2.py` verify:
1. `test_model_v2_shape`: TextRecognizerV2 forward output shape `(T, B, 93)` with 4x horizontal downsampling.
2. `test_model_v2_gradients`: CTC loss backward pass produces real non-zero gradients on linear layers and conv stem.
3. `test_model_v2_param_count`: exactly **5,565,981 parameters** (strictly under the 10M parameter limit).
4. `test_build_recognizer_versions`: factory instantiation of V1 (`TextRecognizerV1`) and V2 (`TextRecognizerV2`).
5. `test_beam_search_ctc_decoding`: CTC beam search decodes target paths with confidence estimation.
6. `test_char_ngram_lm`: multi-order character language model scoring with Laplace smoothing.
7. `test_evaluate_script_test_guard`: evaluation script strictly blocks execution on the test split without `--final-test`.

### Overfit Sanity Tests (Step 1)
- **Single-Sample Memorisation**: Loss dropped from 15.28 to 0.009 in 10 steps (CER = 0.0).
- **20-Sample Real Crops Memorisation Test (`scratch/debug_20_samples.py`)**:
  - Step 1: Loss 18.7834 | CER: 1.0000 | Preds: `['', '', '', '', '']`
  - Step 30: Loss 2.8850 | CER: 0.9403 | Preds: `'t'`, `'t'`, `''`, `'t'`, `'t'`
  - Step 60: Loss 2.0372 | CER: 0.7463 | Preds: `'('.'`, `'L.'`, `'No'`, `'(No'`, `'t.'`
  - Step 80: Loss 1.1226 | CER: 0.5224 | Preds: `'(er'`, `'M.'`, `'1.o'`, `'D.'`, `'T.'`
  - Step 100: Loss 0.4303 | CER: 0.2985 | Preds: `'Paer'`, `'Ms.'`, `'1.5'`, `'DAT:'`, `'T.'`
  - Step 130: Loss 0.1407 | CER: 0.1642 | Preds: `'Paer'`, `'Ms.'`, `'1.5'`, `'DATE:'`, `'T.'`
  - Step 150: Loss 0.0609 | **CER: 0.0448 (< 0.05)** | Preds: `'Paer'`, `'Ms.'`, `'1.5'`, `'DATE:'`, `'T.'`
  - **Result**: PASSED in 150 steps (244.1s). Monotonic CER drop below 0.05 threshold achieved.

### Training & Evaluation Tools
- `train_recognizer_gpu.py`: 3-stage curriculum (A: 100% synthetic, B: 50/50 mix, C: 100% real fine-tune), AdamW, cosine LR, GradScaler AMP, resumable checkpoints (`--resume`), early stopping with patience, CSV and JSON logging.
- `evaluate_recognizer.py`: Granular error breakdowns (synthetic unseen val, real FUNSD val, detector-matched `det_*`, words, lines, length buckets, top 20 character confusions). Guarded with `--final-test`.
- `make_kaggle_bundle.py`: Generates `outputs/kaggle_recognizer_bundle.zip` (161.25 MB, 78,873 files) and [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md).

## Phase 2 update — Recognizer Training Data Pipeline

All 48 tests pass and the source/documentation audit passes (`python check_project.py`).
Six new unit tests in `tests/test_recognition_data.py` verify:
- Dense character coverage across the complete `config.json` alphabet.
- Font family isolation: validation font families (`lora`, `source_code_pro`, `noto_sans`, `noto_serif`, `fira_mono`) are strictly disjoint from training font families.
- Disjoint splits: zero document ID or sample leakage between train, val, and test manifests; test split contains strictly 100% real crops.
- Label encoding correctness and CTC time-step bounds satisfaction (`w // 4 >= minimum_ctc_steps(tokens)`).
- On-the-fly streaming dataset functionality (`StreamingSyntheticDataset`).
- Preview comparison grid integrity.

### Data Generation Metrics

- **Real FUNSD Crops**:
  - `train`: 25,596 crops (ground truth words, lines, and detector-predicted regions on train pages).
  - `val`: 4,346 crops (ground truth words and lines).
  - `test`: 11,336 crops (strictly held out real crops; zero synthetic samples).
- **Synthetic Crops**:
  - `train`: 20,000 samples generated in 89.22s (**224.2 samples/sec** to disk; 5,200+ samples/sec in memory).
  - `val`: 2,000 samples generated in 6.20s (**322.3 samples/sec**) using unseen font families.
- **Combined Recognizer Manifests**:
  - `datasets/recognition_train.jsonl`: 45,596 samples
  - `datasets/recognition_val.jsonl`: 6,346 samples
  - `datasets/recognition_test.jsonl`: 11,336 samples
- **Visual Verification**: Side-by-side 100-sample comparison grid saved to `outputs/recognition_previews/preview_grid.png` (3,060 x 792 px).

## Phase 1 update — Audit, Fix & Cross-Platform Verification

All 42 tests pass and the source/documentation audit passes (`python check_project.py`).
Environment dependencies were standardized in `requirements.txt` to recreate clean virtual environments reliably, including `Pillow>=10.0.0,<=11.3.0` (avoiding lazy PDF JPEG registration issues), `opencv-contrib-python>=4.8.0`, and `pyclipper>=1.4.0`.
The default configuration `config.json` was switched from the initial 3-epoch baseline to the best trained 20-epoch word detector (`models/detector_20epochs.pt`), aligned with `config_words.json`.

## Word detector training (20 epochs)

Official FUNSD images were downloaded and converted automatically: 127 training, 22 validation and 50 test pages. The prepared manifests contain 19,245 / 3,267 / 8,973 word regions respectively; 577 / 47 / 266 of these are ignored (empty transcription or heavily cropped).

Our detector was trained from random weights for 20 epochs at size 512 on CPU.
The best checkpoint is `models/detector_20epochs.pt`. Test metrics are saved at
`outputs/funsd_20epochs_metrics.json`. See [FUNSD_GUIDE.md](FUNSD_GUIDE.md).
On the 50 official test pages, polygon IoU 0.5 evaluation gives:
- **Precision**: 0.77308 (77.31%)
- **Recall**: 0.61468 (61.47%)
- **F1 Score**: 0.68484 (**0.685**)
- **Counts**: 5,352 true positives, 1,571 false positives, 3,355 false negatives.

This improves substantially over the initial 3-epoch baseline (precision 0.56135, recall 0.30791, F1 0.39769; 2,681 TP, 2,095 FP, 6,026 FN).

Recognizer and orientation models have not received real-data training. Line detector dataset is prepared but its model training has not been run.

## Separate line detector dataset

Separate derived line manifests exist with 127/22/50 pages and 7,640/1,194/2,963 regions (including ignored regions). Full line model training is documented in [LINE_DETECTION_GUIDE.md](LINE_DETECTION_GUIDE.md).

## Verification Commands

### 1. Windows PowerShell (Primary)

```powershell
python check_env.py
python check_project.py
python -m unittest discover -v
```

### 2. Linux / macOS (Secondary)

```bash
python3 check_env.py
python3 check_project.py
python3 -m unittest discover -v
```
