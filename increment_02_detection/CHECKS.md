# Check our detector and crops

Check labelled masks visually before training. Shrunk masks should remain inside text polygons, ignored areas should not contribute to loss, and letterbox padding should not supply labels. A training batch must produce finite loss and gradients.

On held-out prepared pages, report polygon precision, recall and F1 with the stated matching rule. Include close lines, slanted text, small print, blank pages and noisy scans. Inspect probability maps and crop boundaries to understand failures.

Check missing weights, invalid labels, region limits and crop-pixel limits. Optional line orientation must be tested on labelled 0/180-degree crops separately from full-page orientation.

The included synthetic tests establish code behaviour only. Train on real labels before claiming useful detection. The current detector does not implement learned threshold maps or a pretrained backbone.
