"""Direct execution of our algorithms and trained networks, with no OCR service calls."""
from pathlib import Path
import html
import time
from common import ROOT, read_json, save_json, image_write, sha256
from increment_01_preparation.preparation import pages, prepare
from increment_04_layout.layout import blocks as layout_blocks
from increment_05_tables.tables import extract as extract_tables, html as table_html
from validation import validate_config


def run(source, output, until, config, selected=None, rotation=0):
    validate_config(config)
    if until not in range(1, 6) or rotation not in {0, 90, 180, 270}:
        raise ValueError("Invalid increment or rotation")
    orientation, detector, recognizer = None, None, None
    if config["preparation"]["orientation_checkpoint"]:
        from increment_01_preparation.orientation import Orientation
        orientation = Orientation(ROOT / config["preparation"]["orientation_checkpoint"])
    if until >= 2:
        import torch
        torch.set_num_threads(config["cpu_threads"])
        from increment_02_detection.inference import Detector
        detector = Detector(ROOT / config["detector"]["checkpoint"], config["detector"])
    if until >= 3:
        from increment_03_recognition.inference import Recognizer
        rec_cfg = config.get("recognizer", {})
        recognizer = Recognizer(
            ROOT / rec_cfg["checkpoint"],
            decoder=rec_cfg.get("decoder", "greedy"),
            beam_width=rec_cfg.get("beam_width", 5),
            lm_weight=rec_cfg.get("lm_weight", 0.0),
            lm_corpus_dir=ROOT / rec_cfg.get("lm_corpus_dir", "corpus") if rec_cfg.get("lm_corpus_dir") else None,
        )
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh or empty output folder")
    output.mkdir(parents=True, exist_ok=True)
    document = {"format_version": 1, "until": until, "source": Path(source).name, "input_sha256": None,
                "status": "failed", "config": config, "pages": [], "errors": [],
                "model_sha256": {}}
    for name, model in [("detector", detector), ("recognizer", recognizer)]:
        if model is not None:
            document["model_sha256"][name] = sha256(ROOT / config[name]["checkpoint"])
    if orientation is not None:
        document["model_sha256"]["orientation"] = sha256(ROOT / config["preparation"]["orientation_checkpoint"])
    if detector is not None and config["detector"].get("line_orientation_checkpoint"):
        document["model_sha256"]["line_orientation"] = sha256(ROOT / config["detector"]["line_orientation_checkpoint"])
    metrics = []
    try:
        document["input_sha256"] = sha256(source)
        for number, image, original in pages(source, output / "original_pages", config, selected):
            folder = output / f"page_{number:04d}"
            folder.mkdir()
            page = {"number": number, "original_image": original, "coordinate_frame": "prepared_image_pixels",
                    "regions": [], "lines": [], "blocks": [], "tables": [], "errors": [], "status": "failed"}
            document["pages"].append(page)
            for increment in range(1, until+1):
                start = time.perf_counter()
                old_error_count = len(page["errors"])
                try:
                    if increment == 1:
                        image, page["preparation"] = prepare(image, config["preparation"], orientation, rotation)
                        if image.shape[0]*image.shape[1] > config["max_pixels"]:
                            raise ValueError("Prepared image exceeds pixel limit")
                        path = folder / "prepared.png"
                        image_write(path, image)
                        page.update(prepared_image=str(path.resolve()), width=image.shape[1], height=image.shape[0])
                    elif increment == 2:
                        page["regions"] = detector(image, folder)
                    elif increment == 3:
                        page["lines"] = recognizer(page["regions"])
                        page["blocks"] = [{"id": line["id"], "type": "line", "bbox": line["bbox"], "text": line["text"], "line_ids": [line["id"]]} for line in page["lines"]]
                        for line in page["lines"]:
                            if line["status"] == "failed":
                                page["errors"].append(f"line_{line['id']}: {line['error']}")
                    elif increment == 4:
                        page["blocks"] = layout_blocks(page["lines"], config["layout"]["column_gap"])
                    else:
                        page["tables"] = extract_tables(image, page["lines"], config["tables"]["min_area"])
                        assigned = {line for table in page["tables"] for line in table["assigned_line_ids"]}
                        remaining = [line for line in page["lines"] if line["id"] not in assigned]
                        non_tables = layout_blocks(remaining, config["layout"]["column_gap"])
                        table_blocks = []
                        for table in page["tables"]:
                            table["html"] = table_html(table)
                            table_blocks.append({"type": "table", "table_id": table["id"], "bbox": table["bbox"],
                                                 "text": "\n".join("\t".join(c["text"] for c in table["cells"] if c["row"] == row) for row in range(table["rows"])),
                                                 "line_ids": table["assigned_line_ids"]})
                        # Apply the same column/section ordering to tables and paragraphs.
                        combined = non_tables+table_blocks
                        for index, block in enumerate(combined):
                            block["id"] = index
                        from increment_04_layout.layout import reading_order
                        page["blocks"] = reading_order(combined, config["layout"]["column_gap"])
                    page["completed_increment"] = increment
                    page["text"] = "\n".join(block["text"] for block in page["blocks"] if block["text"])
                    page["status"] = "partial" if page["errors"] else "completed"
                    save_json(folder / f"increment_{increment:02d}.json", page)
                    metrics.append({"page": number, "increment": increment, "seconds": round(time.perf_counter()-start, 6),
                                    "status": "partial" if len(page["errors"]) > old_error_count else "completed"})
                except Exception as error:
                    page["errors"].append(f"increment_{increment}: {type(error).__name__}: {error}")
                    metrics.append({"page": number, "increment": increment, "seconds": round(time.perf_counter()-start, 6), "status": "failed"})
                    break
            page["text"] = "\n".join(block["text"] for block in page["blocks"] if block["text"])
            page["status"] = "partial" if page["errors"] and page["text"] else ("failed" if page["errors"] else (
                "empty" if until >= 3 and not page["text"] else "completed"))
        states = [p["status"] for p in document["pages"]]
        document["status"] = "failed" if not states or all(s == "failed" for s in states) else (
            "partial" if any(s in {"partial", "failed"} for s in states) else "completed")
    except Exception as error:
        document["errors"].append(f"{type(error).__name__}: {error}")
        document["status"] = "partial" if document["pages"] else "failed"
    save_json(output / "result.json", document)
    save_json(output / "metrics.json", metrics)
    (output / "text.txt").write_text("\n\f\n".join(p.get("text", "") for p in document["pages"]), encoding="utf-8")
    markdown = ["# Extracted document", ""]
    for page in document["pages"]:
        markdown.extend([f"## Page {page['number']}", "", html.escape(page.get("text", "")), ""])
    (output / "document.md").write_text("\n".join(markdown), encoding="utf-8")
    return document
