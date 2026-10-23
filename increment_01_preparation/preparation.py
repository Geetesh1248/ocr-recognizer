from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from common import image_write
from geometry import crop_quad, rotate_expanded


def pages(source, destination, config, selected=None):
    source = Path(source)
    if not source.is_file() or source.stat().st_size > config["max_bytes"]:
        raise ValueError("Missing/oversized input")
    if source.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        raise ValueError("Supported inputs: PDF, PNG, JPEG, TIFF, BMP")
    destination.mkdir(parents=True, exist_ok=True)
    pdf = source.suffix.lower() == ".pdf"
    if pdf:
        import pypdfium2 as pdfium
        document = pdfium.PdfDocument(str(source))
        count = len(document)
    else:
        document = Image.open(source)
        count = getattr(document, "n_frames", 1)
    try:
        if not 1 <= count <= config["max_pages"]:
            raise ValueError("Document page limit exceeded")
        chosen = list(range(1, count+1)) if selected is None else selected
        if not chosen or len(set(chosen)) != len(chosen) or any(n < 1 or n > count for n in chosen):
            raise ValueError("Invalid/duplicate selected page number")
        for number in chosen:
            if pdf:
                page = document[number-1]
                scale = config["dpi"]/72
                if np.prod(page.get_size())*scale*scale > config["max_pixels"]:
                    page.close()
                    raise ValueError("Rendered PDF page exceeds pixel limit")
                bitmap = page.render(scale=scale)
                pil = bitmap.to_pil().copy()
                bitmap.close()
                page.close()
            else:
                document.seek(number-1)
                if document.width*document.height > config["max_pixels"]:
                    raise ValueError("Image exceeds pixel limit")
                pil = ImageOps.exif_transpose(document.copy())
            rgba = pil.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, "white")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
            image = np.asarray(rgb)[:, :, ::-1].copy()
            original = destination / f"page_{number:04d}.png"
            image_write(original, image)
            yield number, image, str(original.resolve())
    finally:
        document.close()


def deskew_angle(image, limit=7, step=.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 800:
        gray = cv2.resize(gray, (800, max(1, round(gray.shape[0]*800/gray.shape[1]))))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if np.count_nonzero(ink) < 10:
        return 0.
    best_angle, best_score = 0., -1.
    for angle in np.arange(-limit, limit+step/2, step):
        matrix = cv2.getRotationMatrix2D((ink.shape[1]/2, ink.shape[0]/2), float(angle), 1)
        # Use zero padding for the scoring image: white padding is not text.
        score_image = cv2.warpAffine(ink, matrix, (ink.shape[1], ink.shape[0]), borderValue=0)
        score = float(np.var(score_image.sum(axis=1)))
        if score > best_score+1e-6 or (abs(score-best_score) < 1e-6 and abs(angle) < abs(best_angle)):
            best_angle, best_score = float(angle), score
    return best_angle


def perspective_page(image, return_transform=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        corners = cv2.approxPolyDP(contour, .02*cv2.arcLength(contour, True), True)
        if len(corners) == 4 and cv2.contourArea(corners) >= image.shape[0]*image.shape[1]*.4:
            prepared, _, matrix = crop_quad(image, corners.reshape(4, 2), rotate_tall=False, return_transform=True)
            value = (prepared, corners.reshape(4, 2).tolist())
            return (*value, matrix) if return_transform else value
    return (image, None, np.eye(3)) if return_transform else (image, None)


def prepare(image, settings, orientation=None, rotation=0):
    result, details = image.copy(), {"steps": []}
    details["input_size"] = [image.shape[1], image.shape[0]]
    if rotation:
        result, matrix = rotate_expanded(result, -rotation)
        details["steps"].append({"operation": "requested_rotation", "clockwise": rotation, "matrix": matrix.tolist()})
    if orientation is not None:
        angle, confidence = orientation(result)
        if confidence >= settings["orientation_min_score"]:
            result, matrix = rotate_expanded(result, -angle)
            details["steps"].append({"operation": "learned_orientation", "clockwise": angle, "confidence": confidence, "matrix": matrix.tolist()})
        else:
            details["steps"].append({"operation": "orientation_unchanged_low_score", "confidence": confidence})
    if settings["perspective"]:
        result, corners, matrix = perspective_page(result, return_transform=True)
        details["steps"].append({"operation": "flat_page_perspective", "corners": corners, "matrix": matrix.tolist(), "output_size": [result.shape[1], result.shape[0]]})
    if settings["deskew"]:
        angle = deskew_angle(result, settings["max_deskew_degrees"])
        result, matrix = rotate_expanded(result, angle)
        details["steps"].append({"operation": "deskew", "counterclockwise": angle, "matrix": matrix.tolist()})
    if settings["denoise"]:
        result = cv2.medianBlur(result, 3)
        details["steps"].append({"operation": "median_denoise"})
    if settings["contrast"]:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        details["steps"].append({"operation": "local_contrast"})
    details["coordinate_frame"] = "prepared_image_pixels"
    combined = np.eye(3)
    for step in details["steps"]:
        if "matrix" in step:
            matrix = np.asarray(step["matrix"], dtype=float)
            if matrix.shape == (2, 3):
                matrix = np.vstack([matrix, [0, 0, 1]])
            combined = matrix @ combined
    details["input_to_prepared_matrix"] = combined.tolist()
    details["output_size"] = [result.shape[1], result.shape[0]]
    return result, details
