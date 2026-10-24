"""Run end-to-end OCR visual demo on demo_page.pdf and 2 real FUNSD test documents."""
import os
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np

from common import ROOT, read_json, save_json
from pipeline import run

def annotate_page(image_path: str, page_data: dict, output_path: str):
    """Draw detected bounding boxes, recognized text, and table grids onto the page."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image {image_path}")
    
    overlay = img.copy()
    
    # 1. Draw Table Bounding Boxes & Cells
    tables = page_data.get("tables", [])
    for tbl in tables:
        bx1, by1, bx2, by2 = [int(v) for v in tbl["bbox"]]
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 140, 0), 3)  # Bright blue border
        cols = tbl.get("columns", tbl.get("cols", 0))
        cv2.putText(img, f"TABLE {tbl['id']} ({tbl['rows']}x{cols})", 
                    (bx1, max(18, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)
        # Draw cells
        for cell in tbl.get("cells", []):
            cx1, cy1, cx2, cy2 = [int(v) for v in cell["bbox"]]
            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (255, 220, 150), 1)

    # 2. Draw Word/Line Bounding Boxes & Recognized Text
    lines = page_data.get("lines", [])
    for line in lines:
        x1, y1, x2, y2 = [int(v) for v in line["bbox"]]
        text = line.get("text", "")
        conf = line.get("confidence", 0.0)
        
        # Bounding box in vibrant green
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 40), 2)
        
        if text.strip():
            label = f"{text}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position label box directly above the bbox or inside if near top edge
            label_y2 = y1 - 2 if y1 - th - 4 > 0 else y1 + th + 4
            label_y1 = label_y2 - th - 4
            label_x1 = x1
            label_x2 = min(img.shape[1] - 1, x1 + tw + 6)
            
            # Semi-dark label background for readability
            cv2.rectangle(img, (label_x1, label_y1), (label_x2, label_y2), (20, 20, 20), -1)
            # Yellow text label
            cv2.putText(img, label, (label_x1 + 3, label_y2 - 3), font, font_scale, (0, 240, 255), thickness, cv2.LINE_AA)

    # Blend subtle cell overlay
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    cv2.imwrite(output_path, img)
    return output_path


def main():
    demo_out_dir = ROOT / "outputs" / "demo_results"
    if demo_out_dir.exists():
        shutil.rmtree(demo_out_dir)
    demo_out_dir.mkdir(parents=True, exist_ok=True)
    
    config = read_json(ROOT / "config.json")
    
    test_docs = [
        ("Demo Exam/Document (PDF)", ROOT / "examples" / "demo_page.pdf", "demo_page"),
        ("Real FUNSD Form 1 (Image)", ROOT / "datasets" / "funsd_raw" / "dataset" / "testing_data" / "images" / "82092117.png", "funsd_82092117"),
        ("Real FUNSD Form 2 (Image)", ROOT / "datasets" / "funsd_raw" / "dataset" / "testing_data" / "images" / "82251504.png", "funsd_82251504")
    ]
    
    results_summary = []
    
    for title, doc_path, prefix in test_docs:
        print(f"\n=======================================================")
        print(f"RUNNING OCR PIPELINE ON: {title}")
        print(f"Path: {doc_path}")
        print(f"=======================================================")
        
        doc_temp_out = demo_out_dir / f"pipeline_{prefix}"
        result = run(str(doc_path), str(doc_temp_out), until=5, config=config)
        
        first_page = result["pages"][0]
        prep_img_path = first_page["prepared_image"]
        annotated_path = demo_out_dir / f"{prefix}_annotated.png"
        
        annotate_page(prep_img_path, first_page, str(annotated_path))
        
        # Save structured page summary
        clean_page_summary = {
            "document": doc_path.name,
            "status": result["status"],
            "page_number": first_page["number"],
            "dimensions": {"width": first_page["width"], "height": first_page["height"]},
            "detected_lines_count": len(first_page["lines"]),
            "detected_tables_count": len(first_page.get("tables", [])),
            "text": first_page.get("text", ""),
            "lines": [
                {
                    "id": l["id"],
                    "bbox": l["bbox"],
                    "text": l["text"],
                    "confidence": round(l.get("confidence", 0.0), 4)
                }
                for l in first_page["lines"]
            ],
            "tables": [
                {
                    "id": t["id"],
                    "bbox": t["bbox"],
                    "rows": t["rows"],
                    "columns": t.get("columns", t.get("cols", 0)),
                    "cell_count": len(t["cells"])
                }
                for t in first_page.get("tables", [])
            ]
        }
        
        save_json(demo_out_dir / f"{prefix}_summary.json", clean_page_summary)
        (demo_out_dir / f"{prefix}_extracted.txt").write_text(first_page.get("text", ""), encoding="utf-8")
        
        results_summary.append({
            "title": title,
            "prefix": prefix,
            "annotated_path": str(annotated_path),
            "summary": clean_page_summary
        })
        
        print(f"Annotated visualization saved: {annotated_path}")
        print(f"Detected {len(first_page['lines'])} lines/words and {len(first_page.get('tables', []))} tables.")
        try:
            os.startfile(str(annotated_path))
            print(f"Opened image on screen: {annotated_path.name}")
        except Exception as e:
            print(f"Could not auto-open image: {e}")
    
    print("\nAll 3 demo runs completed successfully!")
    return results_summary

if __name__ == "__main__":
    main()
