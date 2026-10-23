# Read the layout code

`gaps()` merges the horizontal or vertical intervals occupied by text boxes. Empty stretches between merged intervals are candidate whitespace gaps.

`reading_order()` uses these gaps recursively. A sufficiently large horizontal gap separates sections; otherwise a sufficiently large vertical gap can separate columns. It only cuts when every line belongs cleanly to one side. A recursion limit and geometric fallback handle cases where no clean split is found.

`blocks()` receives that ordered list. It compares each line's box height with the median height to guess headings. Other lines can join a paragraph when their horizontal overlap is sufficient and their vertical spacing is close enough.

Each block stores its union rectangle, text and original line IDs. The algorithm does not rewrite recognised characters. Its job is grouping and ordering. Failed/empty lines still retain their source records in the full result.

The configured column-gap threshold is in pixels on the prepared image. Changing render resolution can change suitable gap values; measure reading order on labelled examples at the intended resolution.
