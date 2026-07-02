"""routers/webcam.py — WebSocket endpoint for live webcam feed."""

import base64
import json
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from face_pipeline import process_frame
from utils import frame_to_b64

router = APIRouter(tags=["webcam"])


@router.websocket("/ws/webcam")
async def webcam_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)

            if msg.get("type") != "frame":
                continue

            threshold = float(msg.get("threshold", 0.45))
            b64_data  = msg["data"].split(",", 1)[-1]
            img_bytes = base64.b64decode(b64_data)
            arr       = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(json.dumps({"type": "error", "msg": "bad frame"}))
                continue

            try:
                annotated, results = process_frame(frame, threshold)
                resp = {
                    "type":      "result",
                    "image_b64": frame_to_b64(annotated),
                    "results":   results,
                    "timestamp": time.time(),
                }
            except Exception as exc:
                resp = {"type": "error", "msg": str(exc)}

            await websocket.send_text(json.dumps(resp))

    except WebSocketDisconnect:
        pass
