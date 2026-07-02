"""routers/analyze.py — tracking & analytics endpoints for the Analyze tab.

All recognition events are persisted to SQLite via database.py.
Every endpoint here is prefixed /api/analyze/.
"""

import base64
import json
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

import database as db
from face_pipeline import process_frame
from utils import frame_to_b64

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analyze", tags=["analyze"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode(raw: bytes) -> np.ndarray:
    arr   = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image data.")
    return frame


def _persist_results(session_id: int, results: list, frame_no: Optional[int] = None):
    """Write every detected face to the detections table."""
    for r in results:
        db.log_detection(
            session_id  = session_id,
            person_name = r["name"],
            score       = r["score"],
            bbox        = r.get("bbox"),
            frame_no    = frame_no,
        )


# ── Session management ────────────────────────────────────────────────────────

@router.post("/sessions")
async def create_session(
    source: str = Form(...),
    label:  str = Form(""),
    notes:  str = Form(""),
):
    """Open a new tracking session. Returns {session_id}."""
    sid = db.create_session(source=source, label=label, notes=notes)
    return {"session_id": sid}


@router.patch("/sessions/{session_id}/end")
async def end_session(session_id: int, frame_count: int = 0):
    """Mark a session as ended."""
    db.end_session(session_id, frame_count)
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: int):
    db.delete_session(session_id)
    return {"ok": True}


@router.get("/sessions")
async def list_sessions(limit: int = 50, offset: int = 0):
    return {"sessions": db.get_sessions(limit=limit, offset=offset)}


@router.get("/sessions/{session_id}")
async def get_session(session_id: int):
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


# ── Image analysis (tracked) ──────────────────────────────────────────────────

@router.post("/image")
async def analyze_image(
    file:       UploadFile = File(...),
    threshold:  float      = Form(0.45),
    session_id: int        = Form(...),
):
    """Run recognition on one image and log results to *session_id*."""
    frame = _decode(await file.read())
    try:
        annotated, results = process_frame(frame, threshold)
    except Exception as exc:
        log.exception("process_frame failed")
        raise HTTPException(500, str(exc))

    _persist_results(session_id, results)

    return {
        "image_b64": frame_to_b64(annotated),
        "results":   results,
        "timestamp": time.time(),
    }


# ── WebSocket — live tracked webcam ───────────────────────────────────────────

@router.websocket("/ws/track")
async def track_ws(websocket: WebSocket):
    """
    WebSocket for the Analyze tab's live tracking feed.

    Client sends:
        {"type": "start",  "session_id": 42, "threshold": 0.45}
        {"type": "frame",  "data": "<data-url>", "frame_no": 17}
        {"type": "stop"}

    Server sends:
        {"type": "result",  "image_b64": "…", "results": […], "frame_no": 17}
        {"type": "error",   "msg": "…"}
        {"type": "summary", "persons": […]}  ← on stop
    """
    await websocket.accept()
    log.info("Track WS connected: %s", websocket.client)

    session_id: Optional[int] = None
    threshold   = 0.45
    frame_count = 0

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            t   = msg.get("type")

            if t == "start":
                session_id = int(msg["session_id"])
                threshold  = float(msg.get("threshold", 0.45))
                log.info("Track session %d started via WS.", session_id)

            elif t == "frame":
                if session_id is None:
                    await websocket.send_text(json.dumps(
                        {"type": "error", "msg": "send 'start' first"}))
                    continue

                b64_data  = msg["data"].split(",", 1)[-1]
                img_bytes = base64.b64decode(b64_data)
                frame_no  = msg.get("frame_no", frame_count)
                frame     = _decode(img_bytes)

                try:
                    annotated, results = process_frame(frame, threshold)
                    _persist_results(session_id, results, frame_no=frame_no)
                    frame_count += 1

                    await websocket.send_text(json.dumps({
                        "type":      "result",
                        "image_b64": frame_to_b64(annotated),
                        "results":   results,
                        "frame_no":  frame_no,
                        "timestamp": time.time(),
                    }))
                except Exception as exc:
                    log.exception("Track frame error")
                    await websocket.send_text(json.dumps(
                        {"type": "error", "msg": str(exc)}))

            elif t == "stop":
                if session_id is not None:
                    db.end_session(session_id, frame_count)
                    summary = db.get_person_summary(session_id)
                    await websocket.send_text(json.dumps(
                        {"type": "summary", "persons": summary}))
                break

    except WebSocketDisconnect:
        if session_id is not None:
            db.end_session(session_id, frame_count)
        log.info("Track WS disconnected (session %s).", session_id)


# ── Detections query ──────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/detections")
async def list_detections(
    session_id:  int,
    limit:       int           = 200,
    offset:      int           = 0,
    person_name: Optional[str] = None,
):
    return {
        "detections": db.get_detections(
            session_id, limit=limit, offset=offset, person_name=person_name
        )
    }


# ── Analytics ─────────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/summary")
async def session_summary(session_id: int):
    return {
        "session":    db.get_session(session_id),
        "persons":    db.get_person_summary(session_id),
        "timeline":   db.get_timeline(session_id),
    }


@router.get("/stats")
async def global_stats():
    return db.get_global_stats()


@router.get("/persons/summary")
async def all_persons_summary():
    return {"persons": db.get_person_summary()}
