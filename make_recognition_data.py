"""Generate real and synthetic training and validation data for the text recognizer."""
import argparse
import io
import json
import math
from pathlib import Path
import random
import sys
import time
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
from torch.utils.data import IterableDataset

from common import ROOT, read_json, save_json, sha256, image_read, image_write
from geometry import crop_quad, box_iou, quad, rectangle
from prepare_funsd import transform_polygon
from increment_03_recognition.data import Vocabulary, minimum_ctc_steps

DEFAULT_CONFIG_PATH = ROOT / "config.json"


def normalize_label(text: str, alphabet: str) -> str:
    """Normalize unicode and filter text to allowed alphabet characters."""
    text = unicodedata.normalize("NFC", text)
    # Common OCR/scanned replacements
    text = (text.replace("\u2013", "-")
                .replace("\u2014", "-")
                .replace("\u201c", '"')
                .replace("\u201d", '"')
                .replace("\u2018", "'")
                .replace("\u2019", "'")
                .replace("~", "-")
                .replace("`", "'"))
    allowed = set(alphabet)
    filtered = "".join(ch for ch in text if ch in allowed)
    return " ".join(filtered.split())


class CorpusManager:
    """Generates varied text including book prose, form fields, amounts, dates, and rare symbols."""

    def __init__(self, corpus_dir: Path, config_path: Path = DEFAULT_CONFIG_PATH):
        self.alphabet = read_json(config_path)["recognizer"]["alphabet"]
        self.vocabulary = Vocabulary(self.alphabet)
        self.corpus_dir = Path(corpus_dir)

        # Load literary text files
        words = []
        for txt_file in self.corpus_dir.glob("gutenberg_*.txt"):
            content = txt_file.read_text(encoding="utf-8", errors="ignore")
            for token in content.split():
                clean = normalize_label(token, self.alphabet)
                if clean and len(clean) <= 24:
                    words.append(clean)
        self.prose_words = words if words else ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]

        # Load form entities
        form_json = self.corpus_dir / "form_data.json"
        if form_json.is_file():
            self.form_data = read_json(form_json)
        else:
            self.form_data = {
                "field_labels": ["Date:", "Total:", "Subtotal:", "Invoice #:", "Name:", "Address:", "Amount Due:"],
                "first_names": ["John", "Mary", "David", "Sarah"],
                "last_names": ["Smith", "Johnson", "Williams"],
                "company_suffixes": ["LLC", "Inc.", "Corp."],
                "streets": ["Main St.", "Oak Ave."],
                "cities": ["New York", "Chicago"],
                "states": ["CA", "NY", "TX"]
            }

    def sample_form_field(self) -> str:
        label = random.choice(self.form_data["field_labels"])
        val_type = random.choice(["date", "amount", "code", "name", "id", "status"])
        if val_type == "date":
            val = f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/{random.randint(1980, 2026)}"
        elif val_type == "amount":
            curr = random.choice(["$", "", "EUR "])
            val = f"{curr}{random.randint(1, 9999):,}.{random.randint(0, 99):02d}"
        elif val_type == "code":
            val = f"{random.choice(['PO', 'REF', 'INV', 'ITEM'])}-{random.randint(100, 9999)}"
        elif val_type == "name":
            val = f"{random.choice(self.form_data['first_names'])} {random.choice(self.form_data['last_names'])}"
        elif val_type == "id":
            val = f"{random.randint(10000, 999999)}"
        else:
            val = random.choice(["PAID", "PENDING", "APPROVED", "NET 30", "COMPLETED", "VOID"])
        return f"{label} {val}"

    def sample_text(self) -> str:
        r = random.random()
        if r < 0.15:
            # Short tokens & junk form tokens (1 to 4 chars)
            short_choices = [
                "X", "Y", "N", "1", "0", "-", "/", "#", "@", "$", ":", ".", "?", "!",
                "(X)", "[X]", "( )", "[ ]", "(1)", "[1]",
                "No.", "ID:", "PO#", "Tel:", "Fax:", "St.", "Ave.", "Dr.", "Co.", "Inc.",
                "LLC", "Corp.", "Dept.", "Qty", "Amt", "Tax", "USD", "EUR", "N/A", "TBD",
                "P.O.", "Box", "Ref", "Sub", "Net", "Due", "Zip", "Sr.", "Jr.", "1st", "2nd"
            ]
            text = random.choice(short_choices)
        elif r < 0.40:
            # Single word or formatted number / amount
            if random.random() < 0.6:
                text = random.choice(self.prose_words)
            else:
                num = random.randint(0, 999999)
                text = f"${num:,}.{random.randint(0,99):02d}" if random.random() < 0.5 else str(num)
        elif r < 0.70:
            # 2 to 4 words
            count = random.randint(2, 4)
            text = " ".join(random.choice(self.prose_words) for _ in range(count))
        elif r < 0.90:
            # Form field or line
            text = self.sample_form_field()
        else:
            # Rare punctuation & full alphabet balanced coverage
            rare = ['\\', '|', '_', '^', '@', '#', '$', '%', '&', '*', '+', '=', '<', '>', '[', ']', '{', '}']
            sym = random.choice(rare)
            num = random.randint(10, 999)
            text = f"CODE{sym}{num} ({random.choice(self.prose_words)})"

        # Casing variations
        case_r = random.random()
        if case_r < 0.25:
            text = text.upper()
        elif case_r < 0.45:
            text = text.title()
        elif case_r < 0.55:
            text = text.lower()

        return normalize_label(text, self.alphabet)


class FontManager:
    """Manages font files and ensures family-level isolation between train and validation."""

    def __init__(self, fonts_dir: Path):
        self.fonts_dir = Path(fonts_dir)
        splits_path = self.fonts_dir / "splits.json"
        if not splits_path.is_file():
            raise FileNotFoundError(f"Missing {splits_path}")
        self.splits = read_json(splits_path)

        self.train_fonts = []
        for fam, files in self.splits["train_families"].items():
            for f in files:
                p = self.fonts_dir / f
                if p.is_file():
                    self.train_fonts.append(p)

        self.val_fonts = []
        for fam, files in self.splits["val_families"].items():
            for f in files:
                p = self.fonts_dir / f
                if p.is_file():
                    self.val_fonts.append(p)

        if not self.train_fonts or not self.val_fonts:
            raise ValueError("Font families empty; verify fonts/ directory and splits.json")

    def get_font(self, split: str = "train") -> Path:
        fonts = self.val_fonts if split in ("val", "validation") else self.train_fonts
        return random.choice(fonts)


def render_synthetic_image(text: str, font_path: Path, target_height: int = 32, max_width: int = 1024) -> np.ndarray:
    """Render augmented text image emulating printed and detector crops."""
    # 1. Varied text size & vertical occupancy (small footnotes to large headers)
    occupancy = random.choices([0.45, 0.60, 0.75, 0.88], weights=[0.15, 0.35, 0.35, 0.15])[0]
    canvas_target_h = 64
    font_size = int(round(occupancy * canvas_target_h * random.uniform(0.95, 1.05)))
    font_size = max(16, min(56, font_size))

    try:
        font = ImageFont.truetype(str(font_path), font_size)
        # Variable font weight axis sampling
        if hasattr(font, "get_variation_axes") and hasattr(font, "set_variation_by_axes"):
            axes = font.get_variation_axes()
            if axes:
                coords = []
                for axis in axes:
                    name = axis.get("name", b"").decode("latin1", errors="ignore") if isinstance(axis.get("name"), bytes) else str(axis.get("name", ""))
                    min_v = float(axis.get("minimum", 100))
                    max_v = float(axis.get("maximum", 900))
                    def_v = float(axis.get("default", 400))
                    if "weight" in name.lower() or axis.get("tag") in (b'wght', 'wght'):
                        coords.append(random.uniform(min_v, max_v))
                    else:
                        coords.append(def_v)
                font.set_variation_by_axes(coords)
    except Exception:
        font = ImageFont.load_default()

    # Track letter-spacing
    tracking = random.randint(-1, 3) if random.random() < 0.30 else 0
    stroke = 1 if (random.random() < 0.15 and font_size >= 36) else 0

    # Calculate text dimensions
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)

    if tracking == 0:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        tw = max(1, bbox[2] - bbox[0])
        th = max(1, bbox[3] - bbox[1])
        x_offset = -bbox[0]
        y_offset = -bbox[1]
    else:
        # Measure with tracking
        total_w = 0
        max_h = 0
        for ch in text:
            cb = draw.textbbox((0, 0), ch, font=font, stroke_width=stroke)
            total_w += (cb[2] - cb[0]) + tracking
            max_h = max(max_h, cb[3] - cb[1])
        tw = max(1, total_w)
        th = max(1, max_h)
        x_offset, y_offset = 0, 0

    # Crop modes with SAFE clipping (never obliterates text)
    crop_mode = random.choices(["normal", "loose", "tight", "clipped"], weights=[0.45, 0.30, 0.15, 0.10])[0]
    if crop_mode == "loose":
        pad_top = random.randint(8, 20)
        pad_bottom = random.randint(8, 20)
        pad_left = random.randint(10, 28)
        pad_right = random.randint(10, 28)
    elif crop_mode == "tight":
        pad_top = random.randint(1, 4)
        pad_bottom = random.randint(1, 4)
        pad_left = random.randint(2, 6)
        pad_right = random.randint(2, 6)
    elif crop_mode == "clipped":
        # Safe clipping: clip at most 1-3 pixels into margins or ascenders/descenders
        pad_top = random.randint(-3, 2)
        pad_bottom = random.randint(-3, 2)
        pad_left = random.randint(0, 6)
        pad_right = random.randint(0, 6)
    else:
        pad_top = random.randint(4, 12)
        pad_bottom = random.randint(4, 12)
        pad_left = random.randint(6, 16)
        pad_right = random.randint(6, 16)

    canvas_w = max(30, tw + max(0, pad_left) + max(0, pad_right))
    canvas_h = max(24, th + max(0, pad_top) + max(0, pad_bottom))

    # Background color & paper aging
    bg_val = random.randint(235, 255)
    bg_color = (bg_val, bg_val - random.randint(0, 5), bg_val - random.randint(0, 10))
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    cdraw = ImageDraw.Draw(canvas)

    # Show-through (bleed-through from reverse page side)
    if random.random() < 0.15:
        ghost_val = max(180, bg_val - random.randint(12, 28))
        ghost_color = (ghost_val, ghost_val, ghost_val)
        ghost_text = text[::-1] if len(text) > 3 else "Date Ref"
        cdraw.text((random.randint(0, max(1, canvas_w // 3)), random.randint(0, max(1, canvas_h // 3))),
                   ghost_text, font=font, fill=ghost_color)

    # Ink color
    ink_val = random.randint(0, 45) if random.random() < 0.85 else random.randint(45, 90)
    ink_color = (ink_val, ink_val + random.randint(0, 5), ink_val + random.randint(0, 10))

    # Draw main text
    start_x = max(0, pad_left) + x_offset
    start_y = max(0, pad_top) + y_offset

    if tracking == 0:
        cdraw.text((start_x, start_y), text, font=font, fill=ink_color, stroke_width=stroke)
    else:
        cur_x = start_x
        for ch in text:
            cdraw.text((cur_x, start_y), ch, font=font, fill=ink_color, stroke_width=stroke)
            cb = cdraw.textbbox((0, 0), ch, font=font, stroke_width=stroke)
            cur_x += (cb[2] - cb[0]) + tracking

    # Ruled paper noise, underline, or rare strike-through
    r_line = random.random()
    if r_line < 0.12:
        # Underline
        line_y = min(canvas_h - 2, start_y + th + random.randint(2, 6))
        cdraw.line([(0, line_y), (canvas_w, line_y)], fill=ink_color, width=random.randint(1, 2))
    elif r_line < 0.15:
        # Rare strike-through
        line_y = start_y + th // 2 + random.randint(-2, 2)
        strike_val = min(120, ink_val + random.randint(10, 40))
        cdraw.line([(0, line_y), (canvas_w, line_y)], fill=(strike_val, strike_val, strike_val), width=random.randint(1, 2))

    # Convert to OpenCV image for geometric and noise augmentations
    cv_img = np.array(canvas)

    # Wavy baseline (subtle roller/paper distortion)
    if random.random() < 0.25:
        amp = random.uniform(0.6, 1.4)
        freq = random.uniform(0.04, 0.10)
        phase = random.uniform(0, 2 * np.pi)
        ch_h, ch_w = cv_img.shape[:2]
        y_ind, x_ind = np.indices((ch_h, ch_w), dtype=np.float32)
        y_disp = y_ind + amp * np.sin(x_ind * freq + phase)
        cv_img = cv2.remap(cv_img, x_ind, y_disp, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Toner bleed & photocopy spread (morphological gain / ink starvation)
    if random.random() < 0.25:
        ksize = random.choice([2, 3])
        kernel = np.ones((ksize, ksize), np.uint8)
        if random.random() < 0.70:
            # Toner gain (darkens/fattens black text on white bg)
            cv_img = cv2.erode(cv_img, kernel, iterations=1)
        else:
            # Ribbon fading / ink starvation
            cv_img = cv2.dilate(cv_img, kernel, iterations=1)

    # Small rotation / skew (with border replicate to prevent grey boxes)
    if random.random() < 0.40:
        angle = random.uniform(-3.0, 3.0)
        h, w = cv_img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cv_img = cv2.warpAffine(cv_img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Blur / downscale: natural optical defocus
    if random.random() < 0.35:
        sigma = random.uniform(0.4, 0.9)
        ksize = 3 if sigma < 0.7 else 5
        cv_img = cv2.GaussianBlur(cv_img, (ksize, ksize), sigma)

    # Pixel noise
    if random.random() < 0.30:
        noise = np.random.normal(0, random.uniform(2, 6), cv_img.shape).astype(np.float32)
        cv_img = np.clip(cv_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Contrast variation
    if random.random() < 0.30:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-12, 12)
        cv_img = np.clip(cv_img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # JPEG artifacts
    if random.random() < 0.25:
        quality = random.randint(40, 85)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", cv_img, encode_param)
        cv_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # Scale to target height 32, maintaining aspect ratio (INTER_AREA for clean downsampling)
    h, w = cv_img.shape[:2]
    scale = target_height / float(h)
    new_w = max(4, int(round(w * scale)))
    new_w = min(max_width, ((new_w + 3) // 4) * 4)
    resized = cv2.resize(cv_img, (new_w, target_height), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    # Guarantee CTC length validity: steps = new_w // 4 >= minimum_ctc_steps(tokens)
    vocab = Vocabulary(read_json(DEFAULT_CONFIG_PATH)["recognizer"]["alphabet"])
    tokens = vocab.encode(text)
    required_steps = minimum_ctc_steps(tokens)
    current_steps = new_w // 4

    if current_steps < required_steps:
        # Pad horizontally using border replication to avoid flat grey boxes
        pad_needed = (required_steps - current_steps + 1) * 4
        resized = cv2.copyMakeBorder(resized, 0, 0, 0, pad_needed, borderType=cv2.BORDER_REPLICATE)

    return resized


class StreamingSyntheticDataset(IterableDataset):
    """Infinite streaming synthetic dataset for online training on GPU/Kaggle."""

    def __init__(self, corpus_mgr: CorpusManager, font_mgr: FontManager, split: str = "train", max_samples: int = 1000000):
        self.corpus_mgr = corpus_mgr
        self.font_mgr = font_mgr
        self.split = split
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        while count < self.max_samples:
            text = self.corpus_mgr.sample_text()
            if not text:
                continue
            font_path = self.font_mgr.get_font(self.split)
            img = render_synthetic_image(text, font_path)
            yield img, text
            count += 1


def extract_real_funsd_crops(output_dir: Path, config: dict):
    """Crop words, lines, and detector-predicted regions from FUNSD keeping splits strictly disjoint."""
    output_dir = Path(output_dir).resolve()
    images_dir = output_dir / "images"
    raw_dir = ROOT / "datasets" / "funsd_raw" / "dataset"
    words_dir = ROOT / "datasets" / "funsd_prepared"
    lines_dir = ROOT / "datasets" / "funsd_lines_prepared"
    alphabet = config["recognizer"]["alphabet"]

    if not raw_dir.is_dir() or not words_dir.is_dir() or not lines_dir.is_dir():
        print("FUNSD datasets not found on disk; skipping real crop extraction.")
        return {}

    detector_model = None
    det_ckpt = ROOT / config["detector"]["checkpoint"]
    if det_ckpt.is_file():
        from increment_02_detection.inference import Detector
        try:
            detector_model = Detector(det_ckpt, config["detector"])
        except Exception as e:
            print(f"Notice: Detector not loaded for extra crops: {e}")

    summary = {}
    for split, raw_split in [("train", "training_data"), ("val", "training_data"), ("test", "testing_data")]:
        records = []
        split_img_dir = images_dir / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        crop_idx = 0

        # 1. Ground truth word crops
        transforms_split = words_dir / "transforms" / split
        if transforms_split.is_dir():
            for tr_file in sorted(transforms_split.glob("*.json")):
                doc_id = tr_file.stem
                annot_path = raw_dir / raw_split / "annotations" / f"{doc_id}.json"
                prep_img_path = words_dir / "images" / split / f"{doc_id}.png"
                if not annot_path.is_file() or not prep_img_path.is_file():
                    continue
                prep_img = image_read(prep_img_path)
                h, w = prep_img.shape[:2]
                details = read_json(tr_file)
                matrix = details["input_to_prepared_matrix"]
                annot = read_json(annot_path)

                gt_boxes_for_page = []
                for entity in annot.get("form", []):
                    for word in entity.get("words", []):
                        raw_text = word.get("text", "").strip()
                        box = word.get("box", [])
                        if not raw_text or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                            continue
                        clean = normalize_label(raw_text, alphabet)
                        if not clean:
                            continue
                        poly, vis = transform_polygon([[box[0], box[1]], [box[2], box[1]],
                                                       [box[2], box[3]], [box[0], box[3]]], matrix, w, h)
                        if poly is not None and vis >= 0.85:
                            rect = cv2.minAreaRect(np.array(poly, np.float32))
                            pts = cv2.boxPoints(rect)
                            try:
                                pts = quad(pts)
                                crop, _ = crop_quad(prep_img, pts)
                                crop_idx += 1
                                crop_rel = f"images/{split}/word_{doc_id}_{crop_idx:05d}.png"
                                image_write(output_dir / crop_rel, crop)
                                records.append({"image": crop_rel, "document_id": f"funsd_{split}_{doc_id}", "text": clean})
                                gt_boxes_for_page.append((rectangle(poly), clean))
                            except Exception:
                                pass

                # 2. Train split: Detector crops paired with overlapping ground-truth words
                if split == "train" and detector_model is not None and gt_boxes_for_page:
                    try:
                        import tempfile
                        with tempfile.TemporaryDirectory() as td:
                            regions = detector_model(prep_img, Path(td))
                            for r_item in regions:
                                r_bbox = r_item["bbox"]
                                best_iou = 0.0
                                best_text = None
                                for gt_box, gt_text in gt_boxes_for_page:
                                    iou = box_iou(r_bbox, gt_box)
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_text = gt_text
                                if best_iou >= 0.65 and best_text:
                                    det_crop = image_read(r_item["crop"])
                                    crop_idx += 1
                                    crop_rel = f"images/{split}/det_{doc_id}_{crop_idx:05d}.png"
                                    image_write(output_dir / crop_rel, det_crop)
                                    records.append({"image": crop_rel, "document_id": f"funsd_{split}_{doc_id}", "text": best_text})
                    except Exception as e:
                        print(f"Warning on detector crop extraction: {e}")

        # 3. Ground truth line crops
        line_groups_split = lines_dir / "line_groups" / split
        if line_groups_split.is_dir():
            for lg_file in sorted(line_groups_split.glob("*.json")):
                doc_id = lg_file.stem
                prep_img_path = lines_dir / "images" / split / f"{doc_id}.png"
                if not prep_img_path.is_file():
                    continue
                prep_img = image_read(prep_img_path)
                lines_info = read_json(lg_file)
                for item in lines_info:
                    if item.get("ignore"):
                        continue
                    clean = normalize_label(item.get("text", ""), alphabet)
                    if not clean:
                        continue
                    poly = item.get("polygon")
                    if not poly:
                        continue
                    try:
                        rect = cv2.minAreaRect(np.array(poly, np.float32))
                        pts = quad(cv2.boxPoints(rect))
                        crop, _ = crop_quad(prep_img, pts)
                        crop_idx += 1
                        crop_rel = f"images/{split}/line_{doc_id}_{crop_idx:05d}.png"
                        image_write(output_dir / crop_rel, crop)
                        records.append({"image": crop_rel, "document_id": f"funsd_{split}_{doc_id}", "text": clean})
                    except Exception:
                        pass

        # Write split manifest
        manifest_path = output_dir / f"{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[split] = len(records)
        print(f"Extracted real FUNSD {split}: {len(records)} crops", flush=True)

    return summary


def generate_synthetic_dataset(output_dir: Path, corpus_mgr: CorpusManager, font_mgr: FontManager,
                               train_count: int = 20000, val_count: int = 2000, seed: int = 42):
    """Generate on-disk synthetic image dataset with isolated font families for validation."""
    output_dir = Path(output_dir).resolve()
    random.seed(seed)
    np.random.seed(seed)

    summary = {}
    for split, count in [("train", train_count), ("val", val_count)]:
        records = []
        split_img_dir = output_dir / "images" / split
        split_img_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        print(f"Generating synthetic {split} ({count} samples)...", flush=True)
        for i in range(1, count + 1):
            text = corpus_mgr.sample_text()
            while not text:
                text = corpus_mgr.sample_text()
            font_path = font_mgr.get_font(split)
            img = render_synthetic_image(text, font_path)

            rel_path = f"images/{split}/synth_{split}_{i:07d}.png"
            image_write(output_dir / rel_path, img)
            # Distinct document_id per sample or grouped by font family
            doc_id = f"synth_{split}_{font_path.stem}_{i:07d}"
            records.append({"image": rel_path, "document_id": doc_id, "text": text})

        elapsed = time.perf_counter() - start
        throughput = count / max(0.001, elapsed)
        print(f"Finished synthetic {split}: {count} samples in {elapsed:.2f}s ({throughput:.1f} samples/sec)")

        manifest_path = output_dir / f"{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[split] = {"count": count, "seconds": round(elapsed, 2), "samples_per_sec": round(throughput, 1)}

    return summary


def combine_manifests(real_dir: Path, synth_dir: Path, final_dir: Path):
    """Combine real crops and synthetic samples into standard recognition manifests."""
    import os
    final_dir = Path(final_dir).resolve()
    final_dir.mkdir(parents=True, exist_ok=True)
    datasets_root = ROOT / "datasets"

    counts = {}
    for split in ["train", "val", "test"]:
        combined_records = []

        # Real crops
        real_manifest = real_dir / f"{split}.jsonl"
        if real_manifest.is_file():
            for line in real_manifest.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    abs_img = (real_dir / item["image"]).resolve()
                    rel_to_final = os.path.relpath(abs_img, final_dir).replace("\\", "/")
                    item["image"] = rel_to_final
                    combined_records.append(item)

        # Synthetic samples (only train and val; NEVER test!)
        if split in ("train", "val"):
            synth_manifest = synth_dir / f"{split}.jsonl"
            if synth_manifest.is_file():
                for line in synth_manifest.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        item = json.loads(line)
                        abs_img = (synth_dir / item["image"]).resolve()
                        rel_to_final = os.path.relpath(abs_img, final_dir).replace("\\", "/")
                        item["image"] = rel_to_final
                        combined_records.append(item)

        # Shuffle train split
        if split == "train":
            random.Random(42).shuffle(combined_records)

        out_path = final_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in combined_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = len(combined_records)

        # Also write top-level datasets/recognition_<split>.jsonl for convenience
        top_path = datasets_root / f"recognition_{split}.jsonl"
        top_records = []
        for r in combined_records:
            item = dict(r)
            abs_img = (final_dir / item["image"]).resolve()
            item["image"] = os.path.relpath(abs_img, datasets_root).replace("\\", "/")
            top_records.append(item)
        with top_path.open("w", encoding="utf-8") as f:
            for r in top_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return counts


def generate_preview_grid(synth_dir: Path, real_dir: Path, output_file: Path, num_samples: int = 100):
    """Generate side-by-side visual comparison grid of synthetic vs real FUNSD crops."""
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    synth_manifest = synth_dir / "train.jsonl"
    real_manifest = real_dir / "train.jsonl"
    if not synth_manifest.is_file() or not real_manifest.is_file():
        print("Missing manifests for preview grid.")
        return

    synth_lines = [json.loads(l) for l in synth_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    real_lines = [json.loads(l) for l in real_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

    random.seed(42)
    s_samples = random.sample(synth_lines, min(num_samples, len(synth_lines)))
    r_samples = random.sample(real_lines, min(num_samples, len(real_lines)))

    row_height = 36
    cell_width = 300
    rows = 20
    cols_per_side = 5  # 5 synth columns + 5 real columns = 10 columns total (20 rows x 5 = 100 each)

    grid_width = cell_width * cols_per_side * 2 + 60
    grid_height = row_height * (rows + 2)

    grid = np.full((grid_height, grid_width, 3), 250, dtype=np.uint8)

    # Title headers
    cv2.putText(grid, "SYNTHETIC CROPS (100 Samples)", (cell_width * 2 - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 50, 0), 2)
    cv2.putText(grid, "REAL FUNSD CROPS (100 Samples)", (cell_width * cols_per_side + cell_width * 2 - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 120, 0), 2)

    # Divider line
    divider_x = cell_width * cols_per_side + 30
    cv2.line(grid, (divider_x, 10), (divider_x, grid_height - 10), (160, 160, 160), 2)

    for idx in range(min(num_samples, len(s_samples))):
        c = idx // rows
        r = idx % rows

        # Synthetic cell
        s_item = s_samples[idx]
        s_img = image_read(synth_dir / s_item["image"])
        sh, sw = s_img.shape[:2]
        s_scale = 30.0 / sh
        s_w = min(cell_width - 10, int(round(sw * s_scale)))
        s_resized = cv2.resize(s_img, (s_w, 30))
        y0 = 60 + r * row_height
        x0 = c * cell_width + 10
        grid[y0:y0+30, x0:x0+s_w] = s_resized

        # Real cell
        r_item = r_samples[idx]
        r_img = image_read(real_dir / r_item["image"])
        rh, rw = r_img.shape[:2]
        r_scale = 30.0 / rh
        r_w = min(cell_width - 10, int(round(rw * r_scale)))
        r_resized = cv2.resize(r_img, (r_w, 30))
        x_real = divider_x + 20 + c * cell_width
        grid[y0:y0+30, x_real:x_real+r_w] = r_resized

    image_write(output_file, grid)
    print(f"Generated comparison grid: {output_file}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build recognizer training data (real + synthetic).")
    parser.add_argument("--count", type=int, default=20000, help="Number of synthetic train samples (default 20,000)")
    parser.add_argument("--val-count", type=int, default=2000, help="Number of synthetic validation samples (default 2,000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--skip-real", action="store_true", help="Skip extracting real FUNSD crops if already extracted")
    args = parser.parse_args(argv)

    config = read_json(DEFAULT_CONFIG_PATH)
    corpus_mgr = CorpusManager(ROOT / "corpus", DEFAULT_CONFIG_PATH)
    font_mgr = FontManager(ROOT / "fonts")

    real_dir = ROOT / "datasets" / "recognition_real"
    synth_dir = ROOT / "datasets" / "recognition_synth"
    combined_dir = ROOT / "datasets" / "recognition"

    print("=" * 60)
    print("Recognizer Data Generation Pipeline")
    print("=" * 60)

    # 1. Real FUNSD crops
    if not args.skip_real:
        print("\n[Step 1/4] Extracting real FUNSD crops (words, lines, detector-matched)...")
        real_counts = extract_real_funsd_crops(real_dir, config)
    else:
        print("\n[Step 1/4] Skipping real crop extraction.")

    # 2. Synthetic crops
    print(f"\n[Step 2/4] Generating synthetic crops (Train: {args.count}, Val: {args.val_count})...")
    synth_stats = generate_synthetic_dataset(synth_dir, corpus_mgr, font_mgr,
                                             train_count=args.count, val_count=args.val_count, seed=args.seed)

    # 3. Combine manifests
    print("\n[Step 3/4] Combining manifests and verifying splits...")
    counts = combine_manifests(real_dir, synth_dir, combined_dir)
    print(f"Combined datasets: Train={counts['train']}, Val={counts['val']}, Test={counts['test']}")

    # 4. Preview grid
    print("\n[Step 4/4] Creating side-by-side preview grid...")
    preview_path = ROOT / "outputs" / "recognition_previews" / "preview_grid.png"
    generate_preview_grid(synth_dir, real_dir, preview_path, num_samples=100)

    print("=" * 60)
    print("Recognizer training data generation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
