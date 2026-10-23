# Train a separate line detector

The word detector remains available. Line detection uses the same network
architecture, but trains another set of weights with different target polygons.
It is not a switch that turns the existing word weights into line weights.

| Mode | Labels | Checkpoint | Configuration |
|---|---|---|---|
| Words | `datasets/funsd_prepared` | `models/detector_20epochs.pt` | `config_words.json` |
| Lines | `datasets/funsd_lines_prepared` | `models/detector_lines_20epochs.pt` | `config_lines.json` |

Line labels have already been generated in this workspace. No real line checkpoint
is supplied yet: perform the training below. A tiny temporary training test verifies
the implementation, but its weights are discarded.

## Train, evaluate, run

### 1. Windows PowerShell (Primary)

Run these commands from `FULLY SELF MADE` in order:

```powershell
python cli.py --config config_lines.json train-detector --train datasets\funsd_lines_prepared\detection_train.jsonl --val datasets\funsd_lines_prepared\detection_val.jsonl --output models\detector_lines_20epochs.pt --epochs 20 --size 512
```

This starts a new model from random weights. It does not resume the word model.
An epoch visits the line-labelled training pages once. The best validation-loss
checkpoint is saved, and the epoch history is written next to it. Additional runs
start from scratch and replace the named output; use another filename to preserve
an earlier experiment. Do not run two training commands against the same output.

After training finishes:

```powershell
python cli.py --config config_lines.json evaluate-detector --checkpoint models\detector_lines_20epochs.pt --labels datasets\funsd_lines_prepared\detection_test.jsonl --output outputs\funsd_line_test_metrics.json
python cli.py --config config_lines.json run "C:\path\your-document.pdf" --until 2 --output outputs\lines_01
```

Replace the PDF path. Runtime output needs a fresh or empty folder. The word
version remains independently available:

```powershell
python cli.py --config config_words.json run "C:\path\your-document.pdf" --until 2 --output outputs\words_01
```

### 2. Linux / macOS (Secondary)

```bash
# Train line detector
python3 cli.py --config config_lines.json train-detector --train datasets/funsd_lines_prepared/detection_train.jsonl --val datasets/funsd_lines_prepared/detection_val.jsonl --output models/detector_lines_20epochs.pt --epochs 20 --size 512

# Evaluate
python3 cli.py --config config_lines.json evaluate-detector --checkpoint models/detector_lines_20epochs.pt --labels datasets/funsd_lines_prepared/detection_test.jsonl --output outputs/funsd_line_test_metrics.json

# Run detection
python3 cli.py --config config_lines.json run path/to/document.pdf --until 2 --output outputs/lines_01
```

Inspect `page_0001/crops` and `probability_map.png`. Each predicted region records
its annotation level in the JSON output. Settings reject a checkpoint from the
wrong mode when its metadata declares that mode. Evaluation also rejects mixing
known word and line labels. Historical checkpoints without annotation metadata
remain usable with the original configuration, but cannot be verified as a mode.

## How the line labels are made

`prepare_funsd_lines.py` reads the original FUNSD word annotations. Inside each
annotated entity, `increment_02_detection/line_labels.py` groups vertically aligned
words into rows, then splits a row at gaps exceeding three times text height or
at visible vertical rules. Separate entities remain separate so aligned fields
and table cells do not get joined across a page. Multiline entities become
multiple lines. Empty transcriptions remain ignored regions.

It builds each line polygon in the original pixel coordinates and applies the
saved increment-1 matrix. It copies the exact prepared page images into the new
dataset, retaining identical document splits. Original image and annotation hashes
are checked before conversion; the word dataset and weights are never rewritten.

There are 127 training, 22 validation and 50 test pages. Derived line regions number
7,640 / 1,194 / 2,963 (including 577 / 47 / 266 ignored regions). Of the nonignored
regions, 4,055 / 748 / 1,702 contain multiple words. A single-word line is valid.
`line_groups` records member counts, source entity index, text, polygons and flags.
`previews` contains labelled examples; green means used, red means ignored.

**These are automatically derived line labels, not native human-labelled lines.**
Entity boundaries or conservative gap rules can fragment a visual line, and
incorrect source annotations can propagate. This prevents claiming complete
page-wide line ground truth. Scores measure agreement with this derived dataset;
they cannot be directly compared with word-detector F1. Use validation pages for
tuning and reserve the test pages for final evaluation. For stronger evaluation,
eventually use an independently annotated line dataset matching your documents.

To rebuild after deliberately changing preparation, first prepare a new word
dataset, then point this converter at that dataset. Existing output is not overwritten:

```powershell
python prepare_funsd_lines.py --words datasets\funsd_prepared --output datasets\funsd_lines_v2
```

Linux / macOS equivalent:
```bash
python3 prepare_funsd_lines.py --words datasets/funsd_prepared --output datasets/funsd_lines_v2
```
