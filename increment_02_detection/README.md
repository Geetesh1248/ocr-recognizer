# Increment 2: learn where text appears

This increment implements the actual detector network, training labels/loss, polygon processing and crop generation. There is no call to a complete text-detection product.

**Input:** a prepared page. **Output:** a probability-map image, text polygons with scores and straight text crops.

Our [FUNSD workflow](../FUNSD_GUIDE.md) trains on existing **word boxes**, so its
detector produces word regions. It does not merge them into lines. The existing
crop filename prefix `line_` is retained, but does not imply a complete text line.
Line detection would require consistent line-level training labels.

## How our detector works

`TextDetector` is a small encoder/decoder built from convolution, normalization and activation layers. Downsampling learns broader context. Upsampling and skip connections restore location detail. A one-channel head predicts text-region logits at the input canvas resolution.

Training labels are polygons. We shrink them into masks to help adjacent regions stay separate. The loss combines binary cross-entropy with hard negative selection and Dice overlap. The current network predicts one map with a fixed threshold; a learned threshold branch is not implemented.

During inference, our code letterboxes the page, converts logits to probabilities, thresholds them, extracts/expands contours, rejects noise and duplicates, and maps polygons back to prepared-page pixels. Our geometry code turns them into rectangular crops. Very tall crops are rotated horizontally.

An optional two-class CNN corrects upside-down crops. It is trained separately on line images and starts disabled until a checkpoint is available.

## Train and run

Word detection is preserved. A separate line dataset, checkpoint path and runtime
configuration are provided in [LINE_DETECTION_GUIDE.md](../LINE_DETECTION_GUIDE.md).
The line model trains on automatically grouped FUNSD line regions. Both variants
share the architecture; neither reuses the other's weights by default.

See [TRAINING.md](../TRAINING.md) and [DATA_FORMAT.md](../DATA_FORMAT.md). After training `models/detector.pt`:

```powershell
..\.venv\Scripts\python.exe cli.py run "C:\path\paper.pdf" --until 2 --output outputs\detected_01
```

Without that checkpoint, the command clearly stops. Random initial weights are not offered as a useful OCR detector, and no external OCR fallback is used.

## Important files

`model.py` defines our network. `data.py` generates resized images and shrunk masks. `losses.py` defines training loss. `train.py` performs optimization and saves checkpoints. `inference.py` converts maps into crops. `line_orientation.py` loads our optional crop classifier.

[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) explains the code. [CHECKS.md](CHECKS.md) and `diagrams.tex` give testing and visual reference. The next increment reads the crop characters.
