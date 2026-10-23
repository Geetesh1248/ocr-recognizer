"""Configuration checks before running any image or model operations."""
import math


def validate_config(config):
    for key in ["dpi", "max_bytes", "max_pages", "max_pixels", "cpu_threads"]:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    preparation = config["preparation"]
    if not 0 <= preparation["max_deskew_degrees"] <= 15:
        raise ValueError("Deskew range must be between 0 and 15 degrees")
    for key in ["deskew", "perspective", "denoise", "contrast"]:
        if not isinstance(preparation[key], bool):
            raise ValueError(f"{key} must be true or false")
    detector = config["detector"]
    for key in ["pixel_threshold", "region_threshold", "duplicate_iou", "line_orientation_min_score"]:
        if not isinstance(detector[key], (int, float)) or not math.isfinite(detector[key]) or not 0 <= detector[key] <= 1:
            raise ValueError(f"Invalid probability threshold: {key}")
    if not 0 <= preparation["orientation_min_score"] <= 1:
        raise ValueError("Invalid orientation threshold")
    for key in ["min_area", "expand_ratio", "max_regions", "max_crop_pixels"]:
        if not math.isfinite(detector[key]) or detector[key] <= 0:
            raise ValueError(f"Invalid detector bound: {key}")
    if config["layout"]["column_gap"] <= 0 or config["tables"]["min_area"] <= 0:
        raise ValueError("Layout/table thresholds must be positive")
    alphabet = config["recognizer"]["alphabet"]
    if not alphabet or len(set(alphabet)) != len(alphabet):
        raise ValueError("Recognition alphabet must have unique characters")
