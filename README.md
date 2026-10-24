# Fully Self Made OCR

This is a separate project inside `FULLY SELF MADE`. Its OCR networks and processing algorithms are implemented in this folder. There are no calls to a ready-made OCR engine, no external prediction service, and no downloaded pretrained model weights.

PyTorch supplies basic neural-network layers and gradient calculation. OpenCV, Pillow and NumPy supply image operations. PDFium opens PDF pages. Our code connects those basic tools into the actual detector, recognizer, training loops and document-processing algorithms.

## The five increments

Increment 2 has independent word and line training workflows. Follow
[LINE_DETECTION_GUIDE.md](LINE_DETECTION_GUIDE.md) for the line model, or
[FUNSD_GUIDE.md](FUNSD_GUIDE.md) for the word model. Select
`--config config_words.json` or `--config config_lines.json` before the CLI command.
Default `config.json` uses our trained 20-epoch word detector (`models/detector_20epochs.pt`, test F1 0.685).
The line dataset is prepared, but its full training still needs to be run.

1. [Prepare the page](increment_01_preparation/README.md): rendering, colour conversion, our tilt correction, contrast/noise options, flat-page perspective correction, and an optional orientation CNN we train ourselves.
2. [Detect and crop text](increment_02_detection/README.md): our segmentation network, labelled masks, training loss, heatmap processing, polygons and line crops. Optional upside-down crop correction also uses our own trained classifier.
3. [Recognize characters](increment_03_recognition/README.md): our CNN + bidirectional LSTM, CTC training, character vocabulary and decoder.
4. [Arrange the page](increment_04_layout/README.md): our geometric column ordering, paragraph grouping and heading rules.
5. [Reconstruct bordered tables](increment_05_tables/README.md): our line/grid analysis, cell merging and assignment of recognized text to cells.

The first three are the core OCR system. Layout and tables extend it. We deliberately start with printed English academic documents. Curved-page flattening, borderless tables, a learned layout classifier, beam-search decoding and a custom VLM are future extensions rather than disguised calls to someone else's model.

## Run increment 1 now

### Windows PowerShell (Primary)

```powershell
python cli.py run examples\demo_page.pdf --until 1 --output outputs\prepared_01
```

### Linux / macOS (Secondary)

```bash
python3 cli.py run examples/demo_page.pdf --until 1 --output outputs/prepared_01
```

The preparation algorithms work without learned weights. Open `outputs\prepared_01\page_0001\prepared.png`. The bundled PDF is an illustrative example; replace its path with your own PDF when ready.

The optional orientation models start disabled because we have not trained them on your data. Small-angle tilt correction starts enabled. There is no secret replacement with a pretrained model.

## Train before running increments 2 and 3

For automatic dataset preparation, follow [FUNSD_GUIDE.md](FUNSD_GUIDE.md).
It downloads labelled scanned forms and transforms their word boxes alongside
the images. No manual re-annotation is required. Training on these labels makes
increment 2 a word detector; its crops are words rather than complete lines.
Our trained 20-epoch word detector (`models/detector_20epochs.pt`) achieves test F1 0.685 (77.3% precision, 61.5% recall at IoU 0.5).

Later increments need checkpoints saved by our training commands. The FUNSD
workflow supplies real labelled data for the detector; the recognizer and optional
orientation classifiers still need their own training. A missing checkpoint
produces a clear error and exits; the program does not fabricate OCR predictions.

Follow [DATA_FORMAT.md](DATA_FORMAT.md) to supply labelled examples and [TRAINING.md](TRAINING.md) for training commands. Then:

### Windows PowerShell (Primary)
```powershell
python cli.py run "C:\path\paper.pdf" --until 3 --output outputs\basic_ocr
python cli.py run "C:\path\paper.pdf" --until 5 --output outputs\complete_ocr
```

### Linux / macOS (Secondary)
```bash
python3 cli.py run "path/to/paper.pdf" --until 3 --output outputs/basic_ocr
python3 cli.py run "path/to/paper.pdf" --until 5 --output outputs/complete_ocr
```

`text.txt` is the selected text, `document.md` adds page headings, and `result.json` records locations, scores, table cells, errors and model-file hashes. All predictions are produced automatically. Labels are used for training and offline scoring, not for runtime approval.

## What has actually been verified?

See [VALIDATION.md](VALIDATION.md). All 57 unit tests pass (`python -m unittest discover -s tests`) and the code audit passes (`python check_project.py`).
Tests exercise image algorithms, network gradients, CTC decoding, tiny training runs, checkpoint loading, ResNet-BiLSTM V2 recognizer architecture, CTC beam search, character n-gram LM, and the combined processing path.

On the 50 official FUNSD test pages:
- **Trained 20-Epoch Word Detector** (`models/detector_20epochs.pt`):
  - **Precision**: 77.3% (`0.7731`)
  - **Recall**: 61.5% (`0.6147`)
  - **F1 Score**: 68.5% (`0.6848`) at polygon IoU 0.5 (from `outputs/funsd_20epochs_metrics.json`).
- **Trained Three-Stage Text Recognizer** (`models/recognizer.pt`):
  - **Architecture**: `TextRecognizerV2` (5,565,981 parameters, ResNet + 2-layer BiLSTM + CTC loss).
  - **Real FUNSD Validation CER**: **4.13%** (`0.0413`), WER: **16.97%** (`0.1697`).
  - **Final Held-Out Real FUNSD Test CER**: **7.36%** (`0.0736`), WER: **26.33%** (`0.2633`) across 11,336 real test crops (8,651 word crops at **7.09% CER** and 2,685 line crops at **7.61% CER**).
  - Evaluated strictly once under `--final-test` guard (`outputs/test_metrics_final.json`).

Each increment includes a plain-language README, `CHECKS.md`, a Python code walkthrough and a standalone `diagrams.tex`. [PROJECT_GUIDE.md](PROJECT_GUIDE.md) explains the full flow and the root files. [REFERENCES.md](REFERENCES.md) identifies the methods that informed the implementation.
