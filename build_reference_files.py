"""Regenerate standalone visual references for our five implementations."""
from common import ROOT

DATA = [
    ("increment_01_preparation", "Our page preparation", ["Open a local PDF or image", "Render pages and normalize colour/background", "Optional: our trained page-orientation CNN", "Our small-angle projection-based deskew", "Optional flat-page perspective and image cleanup", "Save the prepared image and transform details"],
     "Only the optional orientation classifier needs weights. Deterministic preparation runs immediately. Curved-page flattening is a future extension.",
     ["Labelled images and correction angles", "Our three-convolution classifier", "Cross-entropy and gradient updates", "Validate on separate documents", "Save our best orientation checkpoint"]),
    ("increment_02_detection", "Our trainable text detector", ["Prepared page image", "Letterbox and normalize pixels", "Our encoder, skip connections and decoder", "Text probability map", "Threshold, contours, expand and remove duplicates", "Perspective crops and optional own line classifier"],
     "The detector needs our trained checkpoint. Initial random weights are not a usable OCR result. This baseline uses a fixed threshold, not a learned DB threshold branch.",
     ["Page images and labelled text polygons", "Our shrunk masks and valid-pixel mask", "Our segmentation-network logits", "Balanced binary cross-entropy plus Dice loss", "Validate and save the best checkpoint"]),
    ("increment_03_recognition", "Our character recognizer", ["Our saved line crops", "Resize height and retain sequence lengths", "Our convolutional feature extractor", "Our bidirectional LSTM sequence", "Our linear character scores", "Our CTC decoder returns text"],
     "Blank tokens separate repeated letters. Padded positions are excluded from the sequence. Decoder scores are not calibrated accuracy probabilities.",
     ["Line images with complete text labels", "Our vocabulary and CTC-length checks", "Our CNN, LSTM and character layer", "CTC loss and gradient updates", "Validation CER selects our checkpoint"]),
    ("increment_04_layout", "Our reading-order algorithm", ["Recognized lines with pixel rectangles", "Find whitespace intervals", "Recursively divide sections and columns", "Order the line groups", "Group nearby lines into paragraphs", "Mark likely headings and retain line IDs"],
     "This is our geometric implementation. It does not call a document-layout model. Difficult overlapping layouts may need a learned classifier later.",
     ["Existing page and reading-order labels", "Run the geometric rules", "Compare actual order and paragraph groups", "Tune only on validation documents", "Measure final performance on held-out pages"]),
    ("increment_05_tables", "Our bordered-table reconstruction", ["Prepared page and recognized lines", "Morphology extracts horizontal and vertical strokes", "Projection peaks define rows and columns", "Check local separators and rectangular merged cells", "Assign text only when it fits a cell", "Keep ambiguous lines and export cells and HTML"],
     "The first scope is tables with visible borders. Borderless inference is a later extension. Ambiguous text remains in the document rather than being discarded.",
     ["Existing cell and merged-span labels", "Run line and grid reconstruction", "Compare cell boundaries, spans and assignments", "Measure wrong and unassigned cells", "Keep limitations with the automatic result"])
]

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\setlength{\parindent}{0pt}
\setlength{\parskip}{8pt}
\tikzset{step/.style={draw=blue!60!black,fill=blue!5,rounded corners,text width=10cm,align=center,minimum height=13mm,font=\small},arrow/.style={-{Latex},thick}}
\begin{document}
"""


def flow(labels):
    lines = [r"\begin{center}\begin{tikzpicture}[node distance=8mm]"]
    for index, label in enumerate(labels):
        position = "" if index == 0 else f",below=of n{index-1}"
        lines.append(rf"\node[step{position}] (n{index}) {{{label}}};")
        if index:
            lines.append(rf"\draw[arrow] (n{index-1}) -- (n{index});")
    return lines+[r"\end{tikzpicture}\end{center}"]


if __name__ == "__main__":
    for index, (folder, title, runtime, note, development) in enumerate(DATA, 1):
        lines = [PREAMBLE, rf"\begin{{center}}\LARGE\bfseries Increment {index}: {title}\end{{center}}",
                 r"\textbf{Automatic runtime path}"] + flow(runtime) + [note,
                 "All coordinates refer to pixels on the prepared image. See the README and code walkthrough for examples.",
                 r"\newpage", rf"\begin{{center}}\LARGE\bfseries Increment {index}: development\end{{center}}"] + flow(development) + [
                 "Development uses labelled data; runtime does not require approval of predictions. Keep train, validation and final test documents separate.",
                 "Software checks and tiny synthetic training runs do not establish real-document accuracy.",
                 r"\vfill\textit{Fully Self Made OCR: architectures and algorithms implemented in this project.}", r"\end{document}"]
        (ROOT / folder / "diagrams.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Generated five standalone two-page LaTeX references")
