# Read the preparation code

Start with `pages()` in `preparation.py`. It validates the file, opens PDF pages or image frames, converts them into RGB images over a white background, saves an original page image and yields one page at a time. Yielding avoids holding every rendered page in memory.

`prepare()` is the sequence of processing options. It starts with a copy so the original is retained. Each applied transform is recorded in its `steps` list. The final coordinate frame is the prepared image.

`deskew_angle()` turns the image into a binary ink map. It tries small rotations and sums ink across each row. It chooses the angle with the largest row-projection variance: text arranged in horizontal lines tends to create stronger peaks. Blank pages return zero. This is an interpretable heuristic and can fail on graphics or unusual layouts.

`perspective_page()` finds edges, approximates contours by polygons and looks for four document corners covering enough of the image. `geometry.crop_quad()` orders the corners and calculates a mapping into a rectangular destination. Page correction does not apply the tall-line rotation used for text crops.

`OrientationCNN` is our three-convolution feature extractor followed by a four-class linear layer. Its training loop uses labelled correction angles and cross-entropy. `Orientation.__call__()` runs the saved model; `prepare()` applies its answer only above the configured score threshold. The score is not a guarantee of correctness.

`rotate_expanded()` computes a rotation matrix and a larger destination canvas so the page is not cut off. It records that matrix for inspection. Perspective correction records detected corners; the current outputs do not supply a general inverse mapping for original PDF coordinates.
