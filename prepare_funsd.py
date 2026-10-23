"""Convert official FUNSD words to prepared images and detector manifests."""
import argparse
import json
from pathlib import Path
import random
import shutil
import urllib.request
import zipfile
import cv2
import numpy as np
from common import ROOT, image_read, image_write, read_json, save_json, sha256, check_splits, manifest
from increment_01_preparation.preparation import prepare

URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"


def transform_polygon(points, matrix, width, height):
    """Apply the image homography, then clip to the visible prepared canvas."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Invalid source polygon")
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix).T
    if np.any(homogeneous[:, 2] <= 1e-8):
        raise ValueError("Polygon crosses the perspective horizon")
    p = (homogeneous[:, :2] / homogeneous[:, 2:]).astype(np.float32)
    canvas = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
    original_area = abs(cv2.contourArea(p))
    area, clipped = cv2.intersectConvexConvex(p, canvas)
    if clipped is None or area < 1 or original_area < 1:
        return None, 0.
    return clipped.reshape(-1, 2).tolist(), min(1., area/original_area)


def download(raw):
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "dataset.zip"
    if not archive.exists():
        temporary = raw / "dataset.zip.part"
        print("Downloading official FUNSD dataset...", flush=True)
        try:
            with urllib.request.urlopen(URL, timeout=60) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            destination = (raw / item.filename).resolve()
            if not destination.is_relative_to(raw.resolve()):
                raise ValueError("Unsafe dataset archive path")
        zipped.extractall(raw)
    return raw / "dataset"


def convert(dataset, output, config, seed=42, rotation=0):
    from validation import validate_config
    validate_config(config)
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty; do not mix dataset versions")
    training = sorted((dataset / "training_data/annotations").glob("*.json"))
    testing = sorted((dataset / "testing_data/annotations").glob("*.json"))
    if len(training) < 2 or not testing:
        raise ValueError("Expected FUNSD dataset/training_data and testing_data, each with annotations and images")
    random.Random(seed).shuffle(training)
    count = max(1, round(len(training)*.15))
    splits = {"train": training[count:], "val": training[:count], "test": testing}
    orientation = None
    if config["preparation"].get("orientation_checkpoint"):
        from increment_01_preparation.orientation import Orientation
        orientation = Orientation(ROOT / config["preparation"]["orientation_checkpoint"])
    output.mkdir(parents=True, exist_ok=True)
    summary = {"source": URL, "annotation_level": "word", "seed": seed, "config": config,
               "rotation": rotation, "splits": {}}
    for split, annotations in splits.items():
        records, ignored_count, words = [], 0, 0
        for annotation in annotations:
            source = annotation.parent.parent / "images" / (annotation.stem + ".png")
            image = image_read(source)  # raw pixel coordinates; deliberately no EXIF transpose
            if image.shape[0]*image.shape[1] > config['max_pixels']:
                raise ValueError(f"Source image exceeds configured pixel limit: {source}")
            prepared, details = prepare(image, config["preparation"], orientation, rotation)
            if prepared.shape[0]*prepared.shape[1] > config['max_pixels']:
                raise ValueError(f"Prepared image exceeds configured pixel limit: {source}")
            polygons, ignored = [], []
            dropped = 0
            for entity in read_json(annotation)["form"]:
                for word in entity["words"]:
                    box = word["box"]
                    if len(box) != 4 or not np.isfinite(box).all():
                        raise ValueError(f"Invalid word box in {annotation}")
                    x0, y0, x1, y1 = box
                    if x1 <= x0 or y1 <= y0:
                        dropped += 1
                        continue
                    polygon, visible = transform_polygon([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],
                        details["input_to_prepared_matrix"], prepared.shape[1], prepared.shape[0])
                    if polygon is None:
                        dropped += 1
                        continue
                    polygons.append(polygon)
                    ignored.append(not bool(word.get("text", "").strip()) or visible < .9)
            relative = Path("images") / split / (annotation.stem + ".png")
            image_write(output / relative, prepared)
            if len(records) < 3:
                preview = prepared.copy()
                for polygon, ignore in zip(polygons, ignored):
                    cv2.polylines(preview, [np.round(polygon).astype(np.int32)], True,
                                  (0,0,255) if ignore else (0,160,0), 1)
                image_write(output / 'previews' / split / (annotation.stem+'.png'), preview)
            details.update(source_image=str(source), source_sha256=sha256(source),
                           annotation_sha256=sha256(annotation), dropped_words=dropped)
            save_json(output / "transforms" / split / (annotation.stem + ".json"), details)
            records.append({"image": relative.as_posix(), "document_id": annotation.stem,
                            "polygons": polygons, "ignore": ignored, "annotation_level": "word"})
            ignored_count += sum(ignored)
            words += len(polygons)
        path = output / f"detection_{split}.jsonl"
        path.write_text("".join(json.dumps(record, allow_nan=False)+"\n" for record in records), encoding="utf-8")
        summary["splits"][split] = {"pages": len(records), "words": words, "ignored_words": ignored_count}
        print(split, summary["splits"][split], flush=True)
    loaded = [manifest(output / f"detection_{split}.jsonl") for split in splits]
    for i in range(3):
        for j in range(i+1, 3):
            check_splits(loaded[i], loaded[j])
    save_json(output / "preparation_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch official FUNSD archive")
    parser.add_argument("--raw", default=str(ROOT / "datasets/funsd_raw"))
    parser.add_argument("--output", default=str(ROOT / "datasets/funsd_prepared"))
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--rotation", type=int, choices=[0,90,180,270], default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = Path(args.raw)
    dataset = download(raw) if args.download else raw / "dataset"
    convert(dataset, args.output, read_json(args.config), args.seed, args.rotation)


if __name__ == "__main__":
    main()
