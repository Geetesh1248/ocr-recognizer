"""Our polygon ordering, perspective crop and image rotation calculations."""
import cv2
import numpy as np


def quad(points):
    p = np.asarray(points, dtype=np.float32)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError("A quadrilateral needs four finite points")
    centre = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1]-centre[1], p[:, 0]-centre[0]))]
    p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)
    edges = np.roll(p, -1, axis=0)-p
    crosses = edges[:, 0]*np.roll(edges[:, 1], -1)-edges[:, 1]*np.roll(edges[:, 0], -1)
    if not (np.all(crosses > 1e-4) or np.all(crosses < -1e-4)):
        raise ValueError("Degenerate text polygon")
    return p


def crop_quad(image, points, rotate_tall=True, return_transform=False):
    p = quad(points)
    width = max(2, round(max(np.linalg.norm(p[1]-p[0]), np.linalg.norm(p[2]-p[3]))))
    height = max(2, round(max(np.linalg.norm(p[3]-p[0]), np.linalg.norm(p[2]-p[1]))))
    if width*height > image.shape[0]*image.shape[1]*2:
        raise ValueError("Unreasonable crop dimensions")
    rectangle = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
    matrix = cv2.getPerspectiveTransform(p, rectangle)
    result = cv2.warpPerspective(image, matrix, (width, height), borderValue=(255, 255, 255))
    rotation = 90 if rotate_tall and height/width >= 1.5 else 0
    if rotation:
        result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        matrix = np.array([[0, 1, 0], [-1, 0, width-1], [0, 0, 1]]) @ matrix
    if return_transform:
        return result, rotation, matrix
    return result, rotation


def rotate_expanded(image, angle_ccw):
    height, width = image.shape[:2]
    centre = ((width-1)/2, (height-1)/2)
    matrix = cv2.getRotationMatrix2D(centre, angle_ccw, 1)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(np.ceil(height*sine+width*cosine-1e-8))
    new_height = int(np.ceil(height*cosine+width*sine-1e-8))
    matrix[0, 2] += (new_width-1)/2-centre[0]
    matrix[1, 2] += (new_height-1)/2-centre[1]
    return cv2.warpAffine(image, matrix, (new_width, new_height), borderValue=(255, 255, 255)), matrix


def rectangle(points):
    p = np.asarray(points)
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return intersection/max(1e-9, area_a+area_b-intersection)
