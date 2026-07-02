"""face_detector.py — YOLO face detector (stage 1).

Runs before MediaPipe in the pipeline:

    frame → YOLO (locate faces, get boxes) → crop per box →
    MediaPipe FaceLandmarker (align + oval/eye polygons) → embed → match

YOLO is good at finding *where* faces are, including small/partial/multiple
faces in a frame. MediaPipe's landmarker is good at precise alignment and
landmarks but expects a roughly-frontal, already-localised face — feeding it
full surveillance-style frames directly is what caused missed/unstable
detections. Cropping with YOLO first fixes that and also gives us multi-face
support for free.

Requires:
    pip install ultralytics
And a face-tuned YOLO checkpoint, e.g. yolov12n-face.pt, referenced by
``config.YOLO_WEIGHTS_PATH``.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Module-level singleton ────────────────────────────────────────────────────
# Loading a YOLO checkpoint takes real time (weight deserialisation + device
# placement). Load once per process and reuse, same pattern as
# face_alignment._get_landmarker().
_yolo_model = None


def _get_yolo_model():
    """Return the process-wide YOLO model, building it on first call."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    from ultralytics import YOLO
    from config import YOLO_WEIGHTS_PATH

    root = Path(__file__).parent
    weights_path = root / YOLO_WEIGHTS_PATH

    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO face-detector weights not found at '{weights_path}'.\n"
            "Set config.YOLO_WEIGHTS_PATH to a valid yolov*-face.pt checkpoint."
        )

    log.info("Loading YOLO face detector from '%s' …", weights_path)
    _yolo_model = YOLO(str(weights_path))
    log.info("YOLO face detector loaded.")
    return _yolo_model


def _box_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _pad_clip_box(
    x1: int, y1: int, x2: int, y2: int,
    img_w: int, img_h: int,
    pad_frac: float = 0.15,
) -> Tuple[int, int, int, int]:
    """
    Expand a YOLO box by *pad_frac* of its size on each side (MediaPipe's
    landmarker performs better with a little context around the face rather
    than a tight crop), then clip to image bounds.
    """
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * pad_frac), int(bh * pad_frac)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_w, x2 + pad_x),
        min(img_h, y2 + pad_y),
    )


def detect_faces(
    image: np.ndarray,
    conf: float = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Run YOLO face detection on *image* (BGR).

    Returns a list of ``(x1, y1, x2, y2)`` int boxes — padded slightly for
    downstream MediaPipe alignment and clipped to image bounds — sorted by
    area, largest (most prominent) face first. Empty list if no face found.
    """
    from config import YOLO_CONF_THRESHOLD

    if conf is None:
        conf = YOLO_CONF_THRESHOLD

    model = _get_yolo_model()
    h, w = image.shape[:2]

    results = model(image, conf=conf, verbose=False)

    boxes: List[Tuple[int, int, int, int]] = []
    for result in results:
        for xyxy in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            boxes.append(_pad_clip_box(x1, y1, x2, y2, w, h))

    boxes.sort(key=_box_area, reverse=True)
    return boxes
