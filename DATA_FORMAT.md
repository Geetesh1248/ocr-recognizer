# Label files for our models

For automatic preparation of an existing labelled dataset, follow
[FUNSD_GUIDE.md](FUNSD_GUIDE.md). The FUNSD converter uses word-level polygons;
do not mix word and line annotations without a deliberate evaluation policy.

Training manifests are **JSON Lines** files: one complete JSON object per line. Image paths are relative to that manifest's directory. Every record needs a nonempty `document_id` identifying its source document.

Training and validation must have disjoint document IDs and no identical image bytes. We check both automatically. Keep related pages, writers/templates where appropriate, and augmented duplicates in the same split. The final test split must remain untouched while tuning.

## Detector labels

```json
{"image":"images/page_01.png","document_id":"notes_01","polygons":[[[20,40],[300,40],[300,70],[20,70]]],"ignore":[false]}
```

Polygons locate words, complete text lines or regions, not individual character outlines. Values are pixels in the supplied image. Use the same annotation granularity for training and evaluation. `ignore` has one boolean per polygon and defaults to all false when absent.

Supply **prepared page images and matching polygon labels**. The detector trainer resizes them with letterboxing and transforms labels accordingly. It does not secretly run page preparation on training examples, because that could move content without changing annotations. A blank page can have an empty polygon list.

## Recognizer labels

```json
{"image":"images/line_01.png","document_id":"notes_01","text":"Total marks: 80"}
```

Each image contains a line to read. The label is its exact text. Characters must belong to the configured alphabet. Changing the alphabet changes the output layer, so train a compatible recognizer rather than loading an unrelated checkpoint.

The model normalizes crops to a fixed height, limits width, and pads batches. Very long labels may need a larger width or a different crop. It explicitly checks that CTC has enough time steps, including extra steps needed for repeated letters.

## Orientation labels

```json
{"image":"images/sideways_page.png","document_id":"notes_02","correction_clockwise":90}
```

This is the rotation needed to make the image upright, not the rotation originally applied to it. Page orientation accepts 0, 90, 180 or 270. Crop/line orientation accepts 0 or 180 and is trained separately.

## Whole-document text labels

The offline text evaluator accepts normal JSON:

```json
{"pages":[{"number":1,"text":"Total marks: 80"}]}
```

Include exactly the page numbers in the prediction result. Use the intended reading order. Scores normalize whitespace but preserve case and punctuation. They count character and whitespace-delimited word edits; word scoring needs care for languages without spaces.

`examples` contains format-only records. The referenced images are not bundled, so those files are not a ready-to-train dataset. Supply actual labelled data before training.
