# Check reading order and grouping

Use labelled one-column and two-column documents, titles above columns, indentation and mixed-size headings. Compare the output line-ID sequence with the expected reading order. Count missing, repeated or reversed lines.

Check paragraph groups and heading guesses against annotations. Retain source line IDs so errors can be traced back to detections. Include no-clean-gap examples to establish the limits of the geometric fallback.

Gap thresholds depend on prepared-image resolution. Tune on validation examples and use unchanged final test documents. A learned layout classifier is not implemented; the current rules must not be described as one.
