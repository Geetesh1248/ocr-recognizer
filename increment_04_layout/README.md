# Increment 4: put the text in document order

Our OCR has now found and read lines. This increment uses their locations to group columns, paragraphs and likely headings.

**Input:** recognised lines with prepared-page rectangles. **Output:** ordered blocks containing text and source line IDs.

The implementation is geometric. It has no external layout parser or trained region model hidden inside it.

## What happens?

Our recursive whitespace algorithm first looks for substantial horizontal gaps that separate sections. Where appropriate, it looks for vertical gaps separating columns. It processes each group in order, otherwise falling back to top-to-bottom and left-to-right sorting.

It then groups neighbouring lines with overlapping horizontal positions and reasonable vertical distance. Larger text boxes with relatively few words are marked as likely headings. These are explainable heuristics, not guaranteed document semantics.

For example, two clearly separated columns can be read down the left column and then down the right, rather than alternating lines across the page.

## Run

After training detector and recognizer:

```powershell
..\.venv\Scripts\python.exe cli.py run "C:\path\paper.pdf" --until 4 --output outputs\ordered_01
```

`layout.py` contains whitespace cuts and grouping. [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) explains them. [CHECKS.md](CHECKS.md) defines labelled checks, and `diagrams.tex` illustrates the flow.

Complex overlapping content and headings that connect several columns can defeat the rules. A separately trained region classifier is a future extension if measured failures justify it. The next increment reconstructs bordered tables.
