"""Audit our project, including absence of ready-made OCR/model imports."""
import ast
import json
from pathlib import Path
import re
from common import ROOT, read_json
from validation import validate_config

SKIP = {".venv", "__pycache__", "models", "outputs", "datasets"}
FORBIDDEN = {"paddle", "paddleocr", "paddlex", "easyocr", "pytesseract", "doctr", "surya", "transformers", "torchvision", "requests", "httpx"}
errors = []
for folder in sorted(ROOT.glob("increment_*")):
    for name in ["README.md", "CODE_WALKTHROUGH.md", "CHECKS.md", "diagrams.tex"]:
        if not (folder / name).is_file():
            errors.append(f"Missing {folder.name}/{name}")
    tex = folder / "diagrams.tex"
    if tex.exists():
        text = tex.read_text(encoding="utf-8")
        for environment in ["document", "center", "tikzpicture"]:
            if text.count("\\begin{"+environment+"}") != text.count("\\end{"+environment+"}"):
                errors.append(f"Unbalanced LaTeX environment: {folder.name}")
for file in ROOT.rglob("*"):
    if not file.is_file() or SKIP.intersection(file.relative_to(ROOT).parts):
        continue
    if file.suffix == ".py":
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            if any(name.split(".")[0] in FORBIDDEN for name in names):
                errors.append(f"Forbidden ready-made model/OCR/network import: {file}")
    elif file.suffix == ".json":
        json.loads(file.read_text(encoding="utf-8-sig"))
    elif file.suffix == ".md":
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", file.read_text(encoding="utf-8")):
            if "://" not in target and not target.startswith("#") and not (file.parent / target.split("#")[0]).exists():
                errors.append(f"Broken local link: {file.name}: {target}")
validate_config(read_json(ROOT / "config.json"))
if len(list(ROOT.glob("increment_*"))) != 5:
    errors.append("Expected exactly five processing increment folders")
if errors:
    raise SystemExit("\n".join(errors))
print("PASS: five documented increments, source/JSON checks and no ready-made OCR imports")
print("Static LaTeX checks only; real-data accuracy still requires training and held-out evaluation.")
