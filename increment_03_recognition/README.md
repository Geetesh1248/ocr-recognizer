# Increment 3: learn to read the line characters

This increment supplies our recognizer architecture, labelled crop loading, CTC training and decoding. It converts the line images from increment 2 into text.

**Input:** cropped line images and their page locations. **Output:** recognised text, decoder scores and retained line identities.

## How our recognizer works

The CNN learns visual features. We average feature-map height to make a left-to-right sequence. A two-layer bidirectional LSTM examines that sequence, and a linear layer predicts vocabulary characters plus blank.

Training uses a complete line's text label. CTC loss handles the unknown alignment between image positions and characters; individual character boxes are not needed. We reject crops too narrow to represent their labels, including the extra time steps required for repeated letters.

Our greedy decoder chooses the highest-scoring token at each time step, collapses adjacent repeats and removes blanks. A blank between repeated letters preserves both letters. Beam-search decoding is a future enhancement.

## Train and run

Follow [TRAINING.md](../TRAINING.md). After training both checkpoints:

```powershell
..\.venv\Scripts\python.exe cli.py run "C:\path\paper.pdf" --until 3 --output outputs\recognized_01
```

Open `text.txt` and `result.json`. The recognizer's alphabet is saved in its checkpoint. Unsupported training-label characters produce an explicit error rather than being silently removed.

A low score does not request human approval. All text is produced automatically. An empty or failed crop remains recorded, and scores must be evaluated against held-out labels. No useful accuracy is claimed from untrained or tiny-test weights.

## Code and future data work

`model.py` contains our CNN/LSTM/character layers. `data.py` handles vocabulary, resized crops, padded batches and CTC-length checks. `decoder.py` contains our greedy decoder. `train.py` optimizes the model, selects by validation CER and saves checkpoints. `inference.py` reads crops using that checkpoint.

[ML_TRAINING_PLAN.md](ML_TRAINING_PLAN.md) records remaining real dataset work. [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md), [CHECKS.md](CHECKS.md) and `diagrams.tex` explain and test the implementation. The next increment organizes lines into document order.
