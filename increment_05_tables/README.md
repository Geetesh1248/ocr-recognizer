# Increment 5: reconstruct tables with visible borders

This increment implements line detection, grid reconstruction, cell merging and assignment of recognised text. It starts with bordered tables so the result can be explained and tested.

**Input:** the prepared image and recognised line locations. **Output:** table rows, columns, cells, row/column spans, assigned text and HTML.

## What happens?

We threshold the image and use long horizontal/vertical morphological kernels to isolate border strokes. Connected border areas become table candidates. Projection counts locate row and column separators.

Neighbouring grid cells are joined when their separating border is absent. A rectangular joined area becomes a merged cell. Nonrectangular combinations are flagged and retained as individual cells rather than pretending their shape is understood.

A recognised line is assigned to a cell only when its centre is inside and most of its area fits. Lines spanning cells stay unassigned and remain in the ordinary text output. We do not guess which words belong to which cell.

## Run

After training detector and recognizer:

```powershell
..\.venv\Scripts\python.exe cli.py run "C:\path\paper.pdf" --until 5 --output outputs\tables_01
```

`result.json` retains each grid, cell span, text assignment and ambiguity flag. Table cells are exported into readable plain-text rows and escaped HTML inside the result. Reading order places table blocks with the remaining paragraphs.

`tables.py` contains the actual algorithms. [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md), [CHECKS.md](CHECKS.md) and `diagrams.tex` explain them.

Broken/skewed borders, decorative lines, complex merges and borderless tables can still cause errors. Borderless inference and a learned table model are future extensions. Empty detection does not silently invent a table, and no OCR/VLM service is used.
