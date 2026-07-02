"""app.py — FastAPI dashboard for the HAMFace recognition/authentication system."""

import base64
import json
import logging
import os
import pickle
import sys
import time
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import (
    FastAPI, File, Form, HTTPException, Request,
    UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ── Path bootstrap (resolve face_recognition/ root) ───────────────────────────
ROOT = Path(__file__).parent   # …/face_recognition/
sys.path.insert(0, str(ROOT))

from face_pipeline import process_frame, embed, preprocess_face    # noqa: E402
from model_store import (                                           # noqa: E402
    get_gallery, get_label_map, get_model,
    reload_gallery, reload_label_map,
)
import database as _db                                              # noqa: E402
from routers import analyze as analyze_router                       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("dashboard")

# ── App setup ─────────────────────────────────────────────────────────────────
app       = FastAPI(title="HAMFace Dashboard", version="2.0.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ── Database init (runs once at startup) ──────────────────────────────────────
_db.init_db()
log.info("SQLite tracker DB initialised.")

# ── Mount routers ─────────────────────────────────────────────────────────────
app.include_router(analyze_router.router)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _frame_to_b64(frame_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


def _decode_upload(data: bytes) -> np.ndarray:
    arr   = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image data.")
    return frame


# ── Page routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/enroll", response_class=HTMLResponse)
async def enroll_page(request: Request):
    return templates.TemplateResponse("enroll.html", {"request": request})


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    gallery = get_gallery()
    lmap    = get_label_map()
    return {
        "gallery_size":   len(gallery),
        "known_persons":  list(lmap.values()),
        "model_ready":    True,   # get_model() raises on failure; if we got here it loaded
    }


# ── Recognition endpoints ─────────────────────────────────────────────────────

@app.post("/api/recognize/image")
async def recognize_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.45),
):
    """Upload a JPEG/PNG image; returns the annotated frame + match results."""
    try:
        frame = _decode_upload(await file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        annotated, results = process_frame(frame, threshold)
    except Exception as e:
        log.exception("process_frame failed")
        raise HTTPException(500, str(e))

    return {
        "image_b64": _frame_to_b64(annotated),
        "results":   results,
        "timestamp": time.time(),
    }


@app.post("/api/recognize/video_frame")
async def recognize_video_frame(
    file: UploadFile = File(...),
    threshold: float = Form(0.45),
):
    """Process a single video frame (JPEG bytes from the browser canvas)."""
    return await recognize_image(file=file, threshold=threshold)


# ── Enrollment endpoint ───────────────────────────────────────────────────────

@app.post("/api/enroll/person")
async def enroll_person(
    name: str               = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    Add / extend a person in the gallery.

    For each uploaded image:
    1. Align + crop the face (Tasks API).
    2. Embed with the HAMFace model.
    3. Average new embeddings with any existing ones for this person.
    4. Persist updated ``gallery_avg.pkl`` and ``label_map.npy``.

    No retraining required — the gallery is updated in-place.
    """
    from config import RAW_FACES_DIR, DATASET_ROOT, GALLERY_AVG_PKL, IMAGE_SIZE
    from face_alignment import extract_face

    person_dir = ROOT / RAW_FACES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    saved:           int          = 0
    failed:          list         = []
    new_embeddings:  list         = []

    for upload in files:
        raw = await upload.read()
        try:
            img = _decode_upload(raw)
        except ValueError:
            failed.append(f"{upload.filename} (decode error)")
            continue

        result = extract_face(img)
        if result is None:
            failed.append(f"{upload.filename} (no face detected)")
            continue

        face_bgr, _, __ = result

        # Persist raw face npy for potential future retraining
        gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
        np.save(str(person_dir / f"{uuid.uuid4().hex}.npy"), resized)

        try:
            emb = embed(face_bgr)
            new_embeddings.append(emb)
        except Exception:
            log.exception("Embedding failed for %s", upload.filename)
            failed.append(f"{upload.filename} (embedding error)")
            continue

        saved += 1

    if saved == 0:
        raise HTTPException(400, f"No faces could be processed. Failures: {failed}")

    # ── Update gallery ────────────────────────────────────────────────────────
    gallery  = get_gallery()         # already-loaded cache
    lmap     = get_label_map()       # int → name
    inv_map  = {v: k for k, v in lmap.items()}   # name → int

    if name in inv_map:
        label = inv_map[name]
    else:
        label        = max(lmap.keys(), default=-1) + 1
        lmap[label]  = name
        # Persist updated label map (name → int format on disk)
        inv = {v: k for k, v in lmap.items()}
        np.save(
            str(ROOT / DATASET_ROOT / "label_map.npy"),
            np.array(inv), allow_pickle=True,
        )

    # Merge: average new embeddings with any existing mean embedding
    existing = gallery.get(label)
    if existing is not None:
        all_embs = [existing] + new_embeddings
    else:
        all_embs = new_embeddings

    gallery[label] = np.mean(np.array(all_embs), axis=0)

    pkl_path = ROOT / GALLERY_AVG_PKL
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(gallery, f)

    reload_gallery()
    reload_label_map()

    log.info("Enrolled '%s' (label=%d): %d saved, %d failed.", name, label, saved, len(failed))

    return {
        "name":         name,
        "label":        label,
        "saved":        saved,
        "failed":       failed,
        "gallery_size": len(get_gallery()),
    }


# ── Person listing ────────────────────────────────────────────────────────────

@app.get("/api/persons")
async def list_persons():
    lmap    = get_label_map()
    gallery = get_gallery()
    return {
        "persons": [
            {"id": k, "name": v, "in_gallery": k in gallery}
            for k, v in sorted(lmap.items())
        ]
    }


# ── WebSocket — live webcam feed ──────────────────────────────────────────────

@app.websocket("/ws/webcam")
async def webcam_ws(websocket: WebSocket):
    """
    Receive raw JPEG frames from the browser, run recognition, send back
    annotated JPEG + results as JSON.

    Message format (client → server):
        {"type": "frame", "data": "<data-url>", "threshold": 0.45}

    Message format (server → client):
        {"type": "result", "image_b64": "…", "results": […], "timestamp": …}
        {"type": "error",  "msg": "…"}
    """
    await websocket.accept()
    log.info("WebSocket connected: %s", websocket.client)

    try:
        while True:
            raw_msg = await websocket.receive_text()
            msg     = json.loads(raw_msg)

            if msg.get("type") != "frame":
                continue

            threshold = float(msg.get("threshold", 0.45))
            b64_data  = msg["data"].split(",", 1)[-1]
            img_bytes = base64.b64decode(b64_data)

            try:
                frame = _decode_upload(img_bytes)
            except ValueError as e:
                await websocket.send_text(json.dumps({"type": "error", "msg": str(e)}))
                continue

            try:
                annotated, results = process_frame(frame, threshold)
                resp = {
                    "type":      "result",
                    "image_b64": _frame_to_b64(annotated),
                    "results":   results,
                    "timestamp": time.time(),
                }
            except Exception as e:
                log.exception("process_frame error in WebSocket")
                resp = {"type": "error", "msg": str(e)}

            await websocket.send_text(json.dumps(resp))

    except WebSocketDisconnect:
        log.info("WebSocket disconnected: %s", websocket.client)