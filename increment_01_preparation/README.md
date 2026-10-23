# Increment 1: prepare the page ourselves

This increment turns a PDF/photo into the image our detector will receive. It does not call an OCR preprocessor.

**Input:** a PDF or supported image. **Output:** `prepared.png` and a record of the operations applied.

The preparation record now includes `input_to_prepared_matrix`, a combined 3-by-3
matrix for requested rotation, learned rotation, perspective correction and deskew,
plus the input/output dimensions. It starts at the image passed to `prepare()`;
PDF rendering and EXIF handling happen earlier. The FUNSD converter uses this
matrix to move existing word polygons with the image. See
[FUNSD_GUIDE.md](../FUNSD_GUIDE.md) for automatic dataset preparation.

## What happens?

PDFium renders pages. Pillow handles camera orientation and transparent backgrounds. Our code then performs any requested rotation and searches small angles for the strongest horizontal concentration of ink. That projection-based search is our deskew algorithm.

Optional contrast adjustment uses local contrast enhancement, and optional denoising uses a median filter. Flat-page perspective correction searches for a sufficiently large four-corner document outline and maps it to a rectangle. It keeps the input when no suitable outline is found.

An optional four-class CNN predicts clockwise corrections 0, 90, 180 or 270. Its architecture and trainer are supplied in `orientation.py`, but no pretrained weights are used. Leave its checkpoint null until it has been trained on your orientation labels.

This does not flatten a curved book page. That more difficult feature is a future extension, not a model call hidden here.

## Run now

From `FULLY SELF MADE`:

```powershell
..\.venv\Scripts\python.exe cli.py run "C:\path\paper.pdf" --until 1 --output outputs\prepared_01
```

Preparation works without trained detector or recognizer files. Default deskew is enabled; perspective, denoise, contrast and learned orientation start disabled. Settings are in the new project's `config.json`.

## Code to read

`preparation.py` contains PDF/image handling and the image algorithms. `orientation.py` contains our trainable classifier. Root `geometry.py` contains rotation and perspective calculations. [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) follows those functions.

The next increment finds writing on this prepared image. All later coordinates are pixels on that image. See [CHECKS.md](CHECKS.md) and `diagrams.tex` for validation and visual reference.
