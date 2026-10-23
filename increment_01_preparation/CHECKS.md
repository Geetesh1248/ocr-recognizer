# Check page preparation

Use known upright, sideways and slightly tilted pages. Compare the angle and pixel locations with expected transforms. Blank pages should remain unchanged by tilt estimation. PDF page selection and transparent images should produce the intended page/frame and white background.

For perspective correction, use flat pages with labelled document corners; no-corner images should stay usable. This does not test curved-page flattening because that feature is not implemented.

If training orientation, use separate documents and measure correction-angle accuracy before enabling its checkpoint. Scores should be calibrated on validation examples if used for threshold decisions. Do not accept perfect orientation claims from one synthetic image.
