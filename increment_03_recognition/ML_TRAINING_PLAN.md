# Labelled-data work that still remains

Executable training code now exists for the detector, recognizer and optional orientation classifiers. The remaining work is real dataset preparation, training and evaluation, not adding an approval stage to inference.

1. Start with printed English academic material. Gather or supply permitted labelled examples, record sources, and remove private identifiers where needed.
2. Audit error types: missed lines, wrong characters, wrong angle, mixed column order or incorrect table cells. Train the component responsible for the measured problem.
3. Use checked line crops for recognition and matching page polygons for detection. Orientation labels describe the correction needed to make an image upright.
4. Split by original document and related sources, approximately 70% training, 15% validation and 15% final test. Keep duplicates and augmented copies together. The trainer checks overlapping IDs and identical image bytes, but near-duplicate/template leakage still needs a considered data split.
5. Begin with a pilot. A possible scale is 2,000–5,000 line images and a separate 60–100-page evaluation set, revised after feasibility checks. These are planning estimates, not sufficient-data guarantees or data already supplied.
6. Train one component at a time. The recognizer's CTC loss must be finite and its alphabet must cover labels. The detector must learn distinct regions, not only reduce background loss.
7. Tune using validation examples only. Report recognizer CER/WER, detector precision/recall, full-page errors, column order, table cells, latency and memory on held-out examples.
8. Adopt a model only when automatic end-to-end results improve without unacceptable regression on clean pages. Keep older weights and settings so the comparison can be repeated.

Read [TRAINING.md](../TRAINING.md) for the commands and [DATA_FORMAT.md](../DATA_FORMAT.md) for labels. GPU feasibility, dataset size and final quality targets should be decided from measured pilot runs.

A learned layout classifier, learned curved-page flattening, beam-search decoding, borderless-table inference and VLM training remain separate future extensions. The current implementation does not pretend to contain those features.
