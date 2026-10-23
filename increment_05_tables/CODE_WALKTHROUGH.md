# Read the table code

`extract()` converts the page to a binary ink image. Morphological opening with long horizontal and vertical kernels keeps strokes that could be table borders. Contours of the combined strokes supply candidate table areas.

`clusters()` combines nearby projection peaks so a thick border gives one separator position rather than several. At least two x positions and two y positions are required to form a grid.

The grid starts as individual cells. A union-find structure groups neighbours when their local dividing border is missing. Groups covering a complete rectangle become row/column spans. Groups forming an irregular shape receive a flag and are kept as atomic cells.

For each recognised line, its centre chooses a possible cell. At least 80% of its box must fit inside that cell before assignment. Ambiguous spanning lines are recorded under `unassigned_line_ids`; the pipeline preserves them outside the table rather than losing their text.

`html()` writes rows and cells with explicit `rowspan` and `colspan`. It escapes cell content so scanned markup is not treated as executable HTML.

The root pipeline removes only successfully assigned line IDs from ordinary paragraphs, creates table blocks and applies the same reading-order algorithm to the combined output.
