"""Create a clean, shareable zip archive of the project without heavy datasets, outputs, or checkpoints."""
import argparse
from pathlib import Path
import sys
import zipfile
from common import ROOT

EXCLUDE_DIRS = {
    "datasets",
    "outputs",
    "models",
    ".venv",
    "venv",
    "env",
    "ENV",
    "__pycache__",
    ".git",
    ".gemini",
    ".idea",
    ".vscode",
    ".pytest_cache",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".pt",
    ".pth",
    ".bin",
    ".zip",
    ".tar",
    ".gz",
}


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    # Check if any parent folder is in EXCLUDE_DIRS
    for part in rel.parts[:-1]:
        if part in EXCLUDE_DIRS:
            return False
    if path.is_dir():
        return False
    if rel.name.endswith(".history.json"):
        return False
    if path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return False
    return True


def create_archive(output_path: Path) -> Path:
    output_path = Path(output_path).resolve()
    included_files = []

    for item in sorted(ROOT.rglob("*")):
        if item.is_file() and should_include(item):
            # Don't include the target output zip itself if inside ROOT
            if item.resolve() == output_path:
                continue
            included_files.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in included_files:
            rel = file.relative_to(ROOT)
            zf.write(file, arcname=str(rel).replace("\\", "/"))

    size_kb = output_path.stat().st_size / 1024
    print(f"Created shareable zip: {output_path} ({len(included_files)} files, {size_kb:.1f} KB)")
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Package project into a clean shareable zip archive.")
    parser.add_argument("--output", default="ocr_project_clean.zip", help="Path to destination zip file")
    args = parser.parse_args(argv)
    create_archive(Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
