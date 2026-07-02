"""routers/recognize.py — /api/recognize/* endpoints."""

import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from face_pipeline import process_frame
from utils import frame_to_b64

router = APIRouter(prefix="/api/recognize", tags=["recognize"])


def _decode_image(raw: bytes) -> np.ndarray:
    arr   = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")
    return frame


@router.post("/image")
async def recognize_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.45),
):
    frame = _decode_image(await file.read())
    try:
        annotated, results = process_frame(frame, threshold)
    except Exception as exc:
        raise HTTPException(500, str(exc))

    return {
        "image_b64": frame_to_b64(annotated),
        "results":   results,
        "timestamp": time.time(),
    }


@router.post("/video_frame")
async def recognize_video_frame(
    file: UploadFile = File(...),
    threshold: float = Form(0.45),
):
    """Process a single extracted video frame (JPEG bytes)."""
    return await recognize_image(file=file, threshold=threshold)
