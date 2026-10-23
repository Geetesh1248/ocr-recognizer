# Read the detector code

`TextDetector.forward()` in `model.py` passes the page through three feature blocks and a deeper bottleneck. The decoder upsamples and concatenates corresponding encoder features. The final convolution produces one logit per pixel. These are our layer definitions, with no pretrained backbone.

`letterbox()` in `data.py` fits a page into a square without changing its proportions and pads the rest white. It returns separate horizontal/vertical scale factors to account for rounded resized dimensions. The tensor is RGB in the range -1 to 1.

`masks()` scales the labelled polygons with those same factors. It fills a target mask using inward polygon offsets and marks padding or ignored labels invalid. Training must use labels matching the supplied image; we do not rotate training pages without changing their polygons.

`detection_loss()` first calculates pixel-wise binary cross-entropy. It includes positive pixels and the hardest negative pixels so easy background does not dominate. A masked Dice term encourages overlap with the true text masks.

`train()` reads separate training/validation manifests, checks leakage, builds our network and optimizes it with Adam. Validation runs without gradients. The lowest validation-loss model is saved, and every epoch is recorded. Detection precision/recall must still be measured afterwards.

`polygons_from_map()` in `inference.py` thresholds the map, finds contours, calculates mean region scores, expands accepted contours and fits rotated rectangles. It removes strongly overlapping duplicate rectangles and orders the remaining areas geometrically.

`Detector.__call__()` converts those coordinates back to the prepared page, checks limits and saves a crop for each polygon. The optional line-orientation CNN can rotate a crop by 180 degrees. Each result retains its polygon, rectangle, score, crop path and rotations.
