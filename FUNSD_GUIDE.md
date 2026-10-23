# Train our word detector with FUNSD

Word detection remains independent. For the additional line detector, see
[LINE_DETECTION_GUIDE.md](LINE_DETECTION_GUIDE.md); it uses separate manifests,
configuration and weights. Existing word data and checkpoints are preserved.

FUNSD supplies scanned forms and word boxes. We use these existing annotations;
you do not draw labels or ask another OCR model to generate them. Our neural
network still starts from random weights. With this dataset increment 2 detects
**words**, rather than complete lines. Crop filenames retain their existing
`line_` prefix for compatibility; each crop is a detected region, usually a word.
This is a small research dataset, not a guarantee of accuracy on all documents.

Official source: https://guillaumejaume.github.io/FUNSD/
Download: https://guillaumejaume.github.io/FUNSD/download/
Terms: https://guillaumejaume.github.io/FUNSD/work/
Use is for non-commercial research and education under the dataset's terms.

## One workflow

The downloaded/prepared dataset and our trained 20-epoch word detector
checkpoint (`models/detector_20epochs.pt`) already exist in this workspace.
You can run increment 2 immediately.

The 20-epoch detector's held-out test precision/recall/F1 are **77.3% / 61.5% / 68.5%**
at polygon IoU 0.5 (from `outputs/funsd_20epochs_metrics.json`: precision 0.77308,
recall 0.61468, F1 0.68484; 5,352 true positives, 1,571 false positives, 3,355
false negatives). This improves significantly over the earlier 3-epoch baseline
(precision 56.1%, recall 30.8%, F1 39.8%).

### 1. Windows PowerShell (Primary)

Run from `FULLY SELF MADE`:

```powershell
# Optional: rebuild prepared dataset from raw download
python prepare_funsd.py --download

# Train word detector for 20 epochs (saves best validation loss checkpoint)
python cli.py train-detector --train datasets\funsd_prepared\detection_train.jsonl --val datasets\funsd_prepared\detection_val.jsonl --output models\detector_20epochs.pt --epochs 20 --size 512

# Evaluate on the held-out test split
python cli.py evaluate-detector --checkpoint models\detector_20epochs.pt --labels datasets\funsd_prepared\detection_test.jsonl --output outputs\funsd_20epochs_metrics.json

# Run detection on sample document
python cli.py --config config_words.json run examples\demo_page.pdf --until 2 --output outputs\funsd_demo
```

### 2. Linux / macOS (Secondary)

```bash
# Rebuild dataset
python3 prepare_funsd.py --download

# Train word detector
python3 cli.py train-detector --train datasets/funsd_prepared/detection_train.jsonl --val datasets/funsd_prepared/detection_val.jsonl --output models/detector_20epochs.pt --epochs 20 --size 512

# Evaluate detector
python3 cli.py evaluate-detector --checkpoint models/detector_20epochs.pt --labels datasets/funsd_prepared/detection_test.jsonl --output outputs/funsd_20epochs_metrics.json

# Run detection
python3 cli.py --config config_words.json run examples/demo_page.pdf --until 2 --output outputs/funsd_demo
```

The first command downloads the official archive to `datasets/funsd_raw`, prepares
images with increment 1, and writes ready-to-train manifests. For an existing
download, extract it so `datasets/funsd_raw/dataset/training_data` exists, then run
without `--download`. Conversion refuses a nonempty output folder; to try another
configuration use `--output datasets/funsd_prepared_v2 --config config_v2.json`.
Use that same configuration for processing new documents.

We keep the official test set separate. We deterministically reserve 15% of the
official training documents for validation (seed 42). Document IDs and image
hashes are checked across splits. Do not tune settings against the final test set.

## How labels stay aligned

For each image, increment 1 records a 3-by-3 matrix mapping its input pixel
coordinates into the prepared image. Rotation, perspective correction and deskew
are multiplied in the actual processing order. Contrast and denoising do not
change coordinates. The converter applies the combined matrix to all four corners
of every word box, then clips polygons against the output image. Mostly cropped
words (less than 90% visible) and empty transcriptions are ignored in training;
fully invisible and degenerate boxes are dropped. Counts and source hashes are
saved with the transformation records in `transforms/`.

For FUNSD, loading uses the annotated raw image pixel grid without EXIF rotation.
The transform starts at that grid, not at PDF points or an earlier camera image.
The default configuration enables deskew, while perspective correction, denoising,
contrast and learned orientation remain off. Scanned forms usually do not need
camera perspective correction. No orientation weights are supplied.

## What to inspect

`preparation_summary.json` records the settings and split counts. Each manifest
contains image paths relative to its own folder and transformed word polygons.
The converter also saves three labelled preview images per split under `previews/`:
green outlines are training labels, and red outlines are ignored regions.
The trainer generates its masks automatically and saves the best validation-loss
checkpoint. The test evaluator reports polygon precision, recall and F1 at IoU
0.5. A checkpoint passing a smoke run establishes that code works, not that the
detector is accurate. Use held-out results to judge the model.
