"""Generate shrunk-region training masks directly from polygon labels."""
import cv2
import numpy as np
import pyclipper
import torch
from torch.utils.data import Dataset
from common import image_read


def letterbox(image, size):
    height, width = image.shape[:2]
    ratio = min(size/height, size/width)
    new_width, new_height = max(1, round(width*ratio)), max(1, round(height*ratio))
    resized = cv2.resize(image, (new_width, new_height))
    canvas = np.full((size, size, 3), 255, np.uint8)
    canvas[:new_height, :new_width] = resized
    scales = (new_width/width, new_height/height)
    tensor = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float()/127.5-1
    return tensor, scales, (new_height, new_width)


def offset_polygon(points, distance):
    scale = 100.
    path = np.round(np.asarray(points)*scale).astype(np.int64).tolist()
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    paths = offset.Execute(distance*scale)
    return [np.asarray(p, dtype=np.float32)/scale for p in paths]


def masks(polygons, ignored, size, scales, content_size, shrink=.4):
    target = np.zeros((size, size), np.float32)
    valid = np.zeros_like(target)
    valid[:content_size[0], :content_size[1]] = 1  # padding never supplies training labels
    if len(polygons) != len(ignored):
        raise ValueError("Polygon/ignore counts differ")
    if not all(isinstance(flag, bool) for flag in ignored):
        raise ValueError("Ignore labels must be JSON true/false booleans")
    for points, ignore in zip(polygons, ignored):
        p = np.asarray(points, np.float32)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError("Invalid labelled polygon")
        p *= np.asarray(scales)
        if np.any(p < 0) or np.any(p[:, 0] > content_size[1]) or np.any(p[:, 1] > content_size[0]):
            raise ValueError("Label polygon lies outside its image")
        area, perimeter = abs(cv2.contourArea(p)), cv2.arcLength(p, True)
        if ignore or area < 4 or perimeter <= 0:
            cv2.fillPoly(valid, [np.round(p).astype(np.int32)], 0)
            continue
        distance = area*(1-shrink**2)/perimeter
        shrunk = offset_polygon(p, -distance)
        if not shrunk:
            cv2.fillPoly(valid, [np.round(p).astype(np.int32)], 0)
        for region in shrunk:
            cv2.fillPoly(target, [np.round(region).astype(np.int32)], 1)
    return torch.from_numpy(target[None]), torch.from_numpy(valid[None])


class DetectionDataset(Dataset):
    def __init__(self, records, size=512, augment=False):
        self.records, self.size, self.augment = records, size, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = image_read(record["image"])
        if self.augment:
            # Photometric changes do not invalidate polygon locations.
            gain = np.random.uniform(.8, 1.2)
            image = np.clip(image.astype(float)*gain, 0, 255).astype(np.uint8)
            if np.random.rand() < .2:
                image = cv2.GaussianBlur(image, (3, 3), 0)
        tensor, scales, shape = letterbox(image, self.size)
        polygons = record["polygons"]
        target, valid = masks(polygons, record.get("ignore", [False]*len(polygons)), self.size, scales, shape)
        return tensor, target, valid
