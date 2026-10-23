# Train the networks we implemented

The word detector has been trained on FUNSD in this workspace (`models/detector_20epochs.pt`, test F1 0.685). Recognition and
orientation need their own training. For the separate line detector, follow
[LINE_DETECTION_GUIDE.md](LINE_DETECTION_GUIDE.md). Read
[DATA_FORMAT.md](DATA_FORMAT.md) for label formats.

All training scripts detect GPUs automatically (`--device auto` by default). If `torch.cuda.is_available()` is true, training runs on CUDA and prints the device; otherwise it falls back to CPU.

---

## 1. Train the recognizer independently

Before training the recognizer, build the training data (real FUNSD crops + synthetic streaming/offline samples):
```powershell
# Windows PowerShell (Primary)
python make_recognition_data.py --count 20000 --val-count 2000
```
```bash
# Linux / macOS (Secondary)
python3 make_recognition_data.py --count 20000 --val-count 2000
```

To run training:

### Windows PowerShell (Primary)
```powershell
python cli.py train-recognizer --train datasets\recognition_train.jsonl --val datasets\recognition_val.jsonl --output models\recognizer.pt --epochs 30
```

### Linux / macOS (Secondary)
```bash
python3 cli.py train-recognizer --train datasets/recognition_train.jsonl --val datasets/recognition_val.jsonl --output models/recognizer.pt --epochs 30
```

This trains the CNN, LSTM and character layer from their initial random weights. It uses CTC loss and selects the checkpoint with the best validation character-error rate. History is saved beside the checkpoint. This is not fine-tuning a downloaded OCR model.

First check a very small training subset to ensure labels and crops are valid. Then use separate validation examples to choose settings. Keep the final test labels out of those decisions.

---

## 2. Train the detector independently

### Windows PowerShell (Primary)
```powershell
python cli.py train-detector --train datasets\detection_train.jsonl --val datasets\detection_val.jsonl --output models\detector_20epochs.pt --epochs 20 --size 512
python cli.py evaluate-detector --checkpoint models\detector_20epochs.pt --labels datasets\detection_test.jsonl --output outputs\detection_scores.json
```

### Linux / macOS (Secondary)
```bash
python3 cli.py train-detector --train datasets/detection_train.jsonl --val datasets/detection_val.jsonl --output models/detector_20epochs.pt --epochs 20 --size 512
python3 cli.py evaluate-detector --checkpoint models/detector_20epochs.pt --labels datasets/detection_test.jsonl --output outputs/detection_scores.json
```

Our polygon-to-mask code generates shrunk text targets. The network is trained with balanced binary cross-entropy plus Dice overlap loss. The best validation-loss checkpoint is saved. Measure actual polygon recall/precision afterwards; a smaller pixel loss alone does not prove good detection.

This measures polygon-IoU matches, with greedy one-to-one matching at 0.5 by default. It is a defined metric, not a guarantee that every benchmark uses identical evaluation rules.

---

## 3. Optional orientation training

### Windows PowerShell (Primary)
```powershell
python cli.py train-orientation --train datasets\orientation_train.jsonl --val datasets\orientation_val.jsonl --output models\orientation.pt
python cli.py train-line-orientation --train datasets\line_orientation_train.jsonl --val datasets\line_orientation_val.jsonl --output models\line_orientation.pt
```

### Linux / macOS (Secondary)
```bash
python3 cli.py train-orientation --train datasets/orientation_train.jsonl --val datasets/orientation_val.jsonl --output models/orientation.pt
python3 cli.py train-line-orientation --train datasets/line_orientation_train.jsonl --val datasets/line_orientation_val.jsonl --output models/line_orientation.pt
```

Enable their checkpoint paths in `config.json` after evaluating them. They start disabled because no suitable weights have been trained. The page classifier has four angle classes; the line classifier has two.

---

## 4. Test the full system

### Windows PowerShell (Primary)
```powershell
python cli.py run "C:\path\paper.pdf" --until 5 --output outputs\test_document
python cli.py evaluate-text outputs\test_document\result.json datasets\page_text_labels.json --output outputs\text_scores.json
```

### Linux / macOS (Secondary)
```bash
python3 cli.py run "path/to/paper.pdf" --until 5 --output outputs/test_document
python3 cli.py evaluate-text outputs/test_document/result.json datasets/page_text_labels.json --output outputs/text_scores.json
```

Only the runtime increments are automatic image-to-document steps. Labels and training are development tasks. There are no runtime approval prompts.

---

## What to record

Record data provenance and permissions, split rules, input preprocessing, alphabet, image size, learning rate, batch size, seed, epochs, model-file hash and held-out scores. Checkpoint metadata stores the kind, selected epoch, model state, relevant dimensions/alphabet and train/validation manifest hashes. The history stores per-epoch results.

Training seeds are set, but exact reproducibility across hardware/library versions is not guaranteed. Checkpoints do not implement resumable optimizer state, distributed training or automatic early stopping. They save the best validation checkpoint across the requested epochs.

CPU is suitable for software checks and small experiments. A meaningful dataset run may need GPU resources (automatically utilized when CUDA is detected); estimate time and memory on a pilot before committing to a schedule. Never claim the tiny synthetic tests in this repository are a trained accuracy result.
