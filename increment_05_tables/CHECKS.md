# Check bordered-table reconstruction

Use labelled simple grids, thick borders and rectangular merged headers. Compare detected row/column boundaries, cell spans and text assignment with the annotations.

Include broken borders, decorative lines and text crossing cell separators. Ambiguous/nonrectangular merges need visible flags. A line that cannot be confidently assigned must stay in the ordinary document text rather than disappear.

Test that exported HTML escapes scanned text. Check maximum cell counts and no-table pages. Borderless tables and complicated damaged grids remain outside the initial validated scope.

Report structure accuracy and text accuracy separately. A correct cell rectangle does not prove the recognizer read its numbers correctly.
