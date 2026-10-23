# Check our recognizer

Verify vocabulary coverage and line/text pairs before training. Include repeated letters, spaces, digits and punctuation. Impossible CTC lengths must cause a clear error rather than silent infinite-loss suppression.

Check finite CTC gradients, padding/sequence lengths and decoder collapse behaviour. Train on a tiny subset first to detect data-loader mistakes, then evaluate using separate labelled lines. Report CER/WER by document difficulty and language.

Score complete pages too: good recognition on perfect crops does not prove the detector finds every real line. Keep predictions uncorrected for automatic accuracy measurements. Empty/failed crop identities must remain present.

Tiny training tests save genuine model states to temporary folders but do not produce shipped accuracy-qualified weights. Beam-search decoding and handwriting/multilingual acceptance remain later work.
