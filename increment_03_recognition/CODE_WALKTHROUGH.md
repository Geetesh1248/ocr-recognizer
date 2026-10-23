# Read the recognition code

`Vocabulary` assigns character IDs starting at 1. ID 0 is reserved for CTC blank. It validates labels and applies NFC Unicode normalization. The initial configured alphabet covers printed English letters, digits, spaces and common punctuation.

`line_tensor()` scales a crop to a fixed height, caps its width and makes that width divisible by four. Pixel values become -1 to 1. The CNN halves width twice, so `width // 4` gives its valid output-sequence length.

`RecognitionDataset` loads the crop and exact label, optionally makes photometric changes, and checks the minimum CTC length. Adjacent repeated characters need extra positions for separating blanks. Long labels may require a larger width or a better crop.

`collate()` pads a batch's images white but keeps each original sequence length. `TextRecognizer.forward()` extracts CNN features, averages height and packs the sequences so the LSTM does not process padded positions as extra text. Its output has dimensions time, batch and character classes.

`train()` applies log-softmax and CTC loss, checks finite values, backpropagates and clips gradients. Validation decoding measures character edits. The best validation-CER checkpoint is kept; this does not replace final held-out testing.

`greedy_ctc()` tracks the previous token. Consecutive identical tokens collapse. A blank updates the previous token too, allowing the same letter after a blank to appear again. The decoder score is a geometric mean of selected character probabilities and is not calibrated accuracy.

`Recognizer.__call__()` runs the saved model on each crop and links the answer to its original detection record. Individual failures keep empty text and an error rather than deleting the line.
