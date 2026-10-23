"""Create an illustrative page for preparation, not an OCR training benchmark."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

folder = Path(__file__).resolve().parent
image = np.full((800, 600, 3), 255, np.uint8)
for text, y, scale in [("SELF MADE OCR", 90, 1), ("Q1. Define an operating system.", 160, .6),
                       ("Q2. What is virtual memory?", 210, .6), ("Total marks: 80", 270, .7)]:
    cv2.putText(image, text, (55, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)
for x in [60, 300, 540]:
    cv2.line(image, (x, 360), (x, 600), (0, 0, 0), 2)
for y in [360, 440, 520, 600]:
    cv2.line(image, (60, y), (540, y), (0, 0, 0), 2)
for text, x, y in [("Question", 80, 410), ("Marks", 320, 410), ("Q1", 80, 490), ("40", 320, 490),
                   ("Q2", 80, 570), ("40", 320, 570)]:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
matrix = cv2.getRotationMatrix2D((300, 400), 3, 1)
image = cv2.warpAffine(image, matrix, (600, 800), borderValue=(255, 255, 255))
pil = Image.fromarray(image[:, :, ::-1].copy())
pil.save(folder / "demo_page.png")
pil.save(folder / "demo_page.pdf", resolution=100)
print("Created demo_page.png and demo_page.pdf (illustrative inputs only)")
