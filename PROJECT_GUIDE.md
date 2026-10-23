# How our implementation reads a document

Consider a scanned sentence such as “Total marks: 80.” Before a computer can return those words, it must locate the writing and learn how its character shapes map to text.

## Preparation

Our page code opens the PDF, renders its pages and makes normal colour images. Our small-angle tilt algorithm tries rotations and compares how strongly ink is concentrated into horizontal rows. Flat-page perspective correction uses four document corners. Optional orientation classification uses a small CNN defined here and trained on correction-angle labels.

All following polygons and boxes are in **pixels on the saved prepared image**. They are not PDF points or normalized fractions. When preparing detection training data, labels must refer to the same prepared images. Image changes that move content require corresponding polygon changes.

## Detection

Our encoder/decoder network turns a page into a probability map. It learns from masks generated from labelled text polygons. Shrinking those polygons helps neighbouring text regions stay separate.

Our post-processing thresholds the map, finds regions, expands them, removes very similar detections and creates four-corner outlines. Our crop transformation makes straight line images from those outlines.

## Recognition

Our CNN extracts line features. A bidirectional LSTM processes them as a left-to-right sequence. A linear layer predicts characters and a blank symbol. CTC loss lets training use a complete line's text label without individual character boxes.

Our greedy decoder collapses repeated adjacent predictions and removes blanks. Variable-width inputs are padded for batches, and sequence lengths prevent the LSTM from treating padding as additional text. Impossible CTC labels are rejected rather than hidden behind an infinite-loss workaround.

## Layout and tables

Our reading-order code looks for whitespace gaps and recursively separates sections and columns. Nearby lines are grouped into paragraphs; relative height supplies a simple heading guess.

Our table code detects long horizontal and vertical strokes, reconstructs a grid and checks separators between neighbouring cells. Missing separators can indicate a rectangular merged cell. Ambiguous nonrectangular merges are flagged. Text spanning several cells remains unassigned and is retained in the ordinary text output instead of being guessed into a cell.

## Root Python files

| File | Job |
|---|---|
| `cli.py` | Parses run, training and evaluation commands |
| `pipeline.py` | Connects our five increments and writes document outputs |
| `common.py` | File saving, image reading, labelled manifests, split checks and error-count helpers |
| `geometry.py` | Corner ordering, expanded rotation and perspective cropping |
| `checkpoints.py` | Saves model states and loads only matching trained checkpoint formats |
| `evaluate.py` | Text error scores and polygon-based detection scoring |
| `validation.py` | Checks input settings before processing |
| `build_reference_files.py` | Regenerates the LaTeX reference sources |
| `check_project.py` | Audits sources, links and forbidden ready-made OCR imports |

`model.py` files inside the increments contain the actual neural networks. `data.py`, `losses.py` and `train.py` contain the data preparation, losses and optimization. These are substantive implementations, not JSON wrappers around an external OCR API.

## Limits to explain honestly

The detector is a simple segmentation baseline with fixed-threshold post-processing, not a complete DBNet reproduction. The recognizer implements a compact CRNN-style architecture. Layout is geometric and tables initially need visible borders. Neither detector scores nor recognizer decoder scores are calibrated probabilities of correctness.

Training and labelled testing determine whether these implementations work well for your documents. Having a model architecture and a loadable checkpoint does not establish useful reading accuracy.
