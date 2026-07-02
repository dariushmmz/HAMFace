"""utils.py — shared image helpers used across the dashboard."""

import base64

import cv2
import numpy as np


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """BGR face crop → float32 (1, H, W, 3) ready for the model."""
    from config import IMAGE_SIZE
    gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    arr     = resized.reshape((IMAGE_SIZE, IMAGE_SIZE, 1)).astype(np.float32)
    arr     = np.concatenate([arr] * 3, axis=-1)
    return arr[np.newaxis, ...]


def frame_to_b64(frame_bgr: np.ndarray) -> str:
    """Encode a BGR frame as a base-64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()
