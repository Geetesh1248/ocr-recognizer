import cv2
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw

ROOT = Path(__file__).resolve().parent
fonts_dir = ROOT / "fonts"

font_path = fonts_dir / "Roboto-Variable.ttf"
font = ImageFont.truetype(str(font_path), 36)

# Test variable axis sampling
if hasattr(font, "get_variation_axes") and hasattr(font, "set_variation_by_axes"):
    axes = font.get_variation_axes()
    print("Axes for Roboto:", axes)
    coords = []
    for a in axes:
        min_v = a.get("minimum", 100)
        max_v = a.get("maximum", 900)
        coords.append(random.uniform(min_v, max_v))
    font.set_variation_by_axes(coords)
    print("Successfully set variation axes:", coords)

print("Variable font handling tested successfully!")
