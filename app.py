"""Interactive Web UI for the Self-Made OCR System.
Built with Gradio. Supports drag-and-drop Images & PDFs, Greedy vs Beam+LM decoding,
and full 5-stage pipeline extraction (Preparation -> Detection -> Recognition -> Layout -> Tables).
"""
import os
import shutil
import time
import uuid
from pathlib import Path
import cv2
import gradio as gr
import numpy as np

from common import ROOT, read_json, save_json
from pipeline import run
from run_demo import annotate_page


def process_document(file_obj, decoder_choice, beam_width, lm_weight, until_step):
    if file_obj is None:
        return None, "Please upload or select an image or PDF file.", {}, "<p>No tables detected</p>", "No file provided"

    input_path = file_obj if isinstance(file_obj, str) else file_obj.name
    input_path = Path(input_path).resolve()
    
    if not input_path.exists():
        return None, f"File not found: {input_path}", {}, "", "Error"

    # Unique temporary run folder
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    run_dir = ROOT / "outputs" / "ui_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Base configuration
    config = read_json(ROOT / "config.json")
    
    # Configure Recognizer decoding mode
    is_beam = "Beam" in decoder_choice
    config["recognizer"]["decoder"] = "beam" if is_beam else "greedy"
    config["recognizer"]["beam_width"] = int(beam_width) if is_beam else 1
    config["recognizer"]["lm_weight"] = float(lm_weight) if is_beam else 0.0
    config["recognizer"]["lm_corpus_dir"] = "corpus"

    t0 = time.perf_counter()
    pipeline_out = run_dir / "pipeline_output"
    
    try:
        result = run(str(input_path), str(pipeline_out), until=int(until_step), config=config)
    except Exception as e:
        return None, f"Pipeline Error: {e}", {"error": str(e)}, "", f"Failed: {e}"

    total_time = round(time.perf_counter() - t0, 3)

    if not result.get("pages"):
        return None, "No pages processed.", result, "", f"Completed in {total_time}s (0 pages)"

    first_page = result["pages"][0]
    prep_img_path = first_page.get("prepared_image")
    annotated_path = run_dir / "annotated.png"

    if prep_img_path and Path(prep_img_path).exists():
        annotate_page(prep_img_path, first_page, str(annotated_path))
        annotated_result = str(annotated_path)
    else:
        annotated_result = None

    # Text output
    full_text = first_page.get("text", "")
    if not full_text.strip():
        full_text = "(No text detected or recognition increment not reached)"

    # HTML Tables
    tables_html = ""
    tables = first_page.get("tables", [])
    if tables:
        for tbl in tables:
            tables_html += f"<h4>Table {tbl['id']} ({tbl['rows']} rows x {tbl.get('columns', tbl.get('cols', '?'))} columns)</h4>"
            tables_html += tbl.get("html", "")
            tables_html += "<br/><hr/><br/>"
    else:
        tables_html = "<i>No bordered tables detected on this page.</i>"

    # Execution metrics
    lines_count = len(first_page.get("lines", []))
    blocks_count = len(first_page.get("blocks", []))
    tables_count = len(tables)

    metrics_md = f"""
### Pipeline Run Statistics
- **Input File:** `{input_path.name}`
- **Decoder:** `{config['recognizer']['decoder'].upper()}` (Beam Width: `{config['recognizer']['beam_width']}`, LM Weight: `{config['recognizer']['lm_weight']}`)
- **Total End-to-End Latency:** `{total_time}s`
- **Detected Words/Lines:** `{lines_count}`
- **Layout Blocks:** `{blocks_count}`
- **Bordered Tables:** `{tables_count}`
- **Pipeline Status:** `{result['status'].upper()}`
"""

    # Structured JSON
    structured_json = {
        "status": result["status"],
        "source": input_path.name,
        "metrics": {
            "total_latency_seconds": total_time,
            "detected_lines": lines_count,
            "layout_blocks": blocks_count,
            "tables": tables_count
        },
        "lines": [
            {
                "id": l["id"],
                "bbox": l["bbox"],
                "text": l["text"],
                "confidence": round(l.get("confidence", 0.0), 4)
            }
            for l in first_page.get("lines", [])
        ],
        "tables": tables
    }

    return annotated_result, full_text, structured_json, tables_html, metrics_md


def create_app():
    custom_css = """
    .gradio-container { max-width: 1300px !important; margin: auto; }
    #title-banner { text-align: center; margin-bottom: 20px; }
    """
    
    with gr.Blocks(title="Self-Made OCR System Demo") as demo:
        gr.Markdown(
            """
            # 🔍 Self-Made OCR System (Zero Pretrained Weights)
            ### From-Scratch End-to-End Document Intelligence Pipeline
            **ResNet + BiLSTM + CTC Recognizer (5.57M params) | DBNet Word/Line Detector | Layout & Bordered Table Extraction**
            """,
            elem_id="title-banner"
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                file_input = gr.File(
                    label="📄 Drop Document Here (Image or PDF)",
                    file_types=[".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff"],
                    type="filepath"
                )
                
                with gr.Accordion("⚙️ Decoding & Pipeline Settings", open=True):
                    decoder_choice = gr.Radio(
                        choices=["Greedy CTC (Fastest)", "Beam Search + Char-4gram LM"],
                        value="Greedy CTC (Fastest)",
                        label="Recognition Decoding Mode"
                    )
                    
                    with gr.Row():
                        beam_width = gr.Slider(
                            minimum=2, maximum=20, value=5, step=1,
                            label="Beam Width", visible=False
                        )
                        lm_weight = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                            label="Language Model Weight", visible=False
                        )
                    
                    until_step = gr.Slider(
                        minimum=1, maximum=5, value=5, step=1,
                        label="Pipeline Depth (1: Prep, 2: Detect, 3: Recognize, 4: Layout, 5: Tables)"
                    )
                
                run_btn = gr.Button("🚀 Run Full OCR Pipeline", variant="primary", size="lg")
                
                # Preset Examples
                demo_pdf = str(ROOT / "examples" / "demo_page.pdf")
                funsd1 = str(ROOT / "datasets" / "funsd_raw" / "dataset" / "testing_data" / "images" / "82092117.png")
                funsd2 = str(ROOT / "datasets" / "funsd_raw" / "dataset" / "testing_data" / "images" / "82251504.png")
                
                available_examples = []
                for p in [demo_pdf, funsd1, funsd2]:
                    if Path(p).exists():
                        available_examples.append([p, "Greedy CTC (Fastest)", 5, 0.2, 5])
                
                if available_examples:
                    gr.Examples(
                        examples=available_examples,
                        inputs=[file_input, decoder_choice, beam_width, lm_weight, until_step],
                        label="📌 Quick Load Presets"
                    )

            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.TabItem("🖼️ Annotated Document"):
                        annotated_image = gr.Image(label="Visual Detections & Recognitions", interactive=False)
                    with gr.TabItem("📝 Extracted Text"):
                        extracted_text = gr.Textbox(label="Full Document Text", lines=20, max_lines=35, interactive=False)
                    with gr.TabItem("📊 Extracted Tables"):
                        tables_html = gr.HTML(label="Reconstructed Bordered Tables")
                    with gr.TabItem("📋 Structured JSON"):
                        structured_json = gr.JSON(label="Page JSON Output")
                    with gr.TabItem("⏱️ Latency & Metrics"):
                        metrics_display = gr.Markdown()

        # Dynamic visibility toggle for beam search sliders
        def toggle_decoder_settings(choice):
            is_beam = "Beam" in choice
            return gr.update(visible=is_beam), gr.update(visible=is_beam)

        decoder_choice.change(
            fn=toggle_decoder_settings,
            inputs=[decoder_choice],
            outputs=[beam_width, lm_weight]
        )

        run_btn.click(
            fn=process_document,
            inputs=[file_input, decoder_choice, beam_width, lm_weight, until_step],
            outputs=[annotated_image, extracted_text, structured_json, tables_html, metrics_display]
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
