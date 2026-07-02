"""person_store.py — enroll new persons and manage the gallery + label map."""

import logging
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import UploadFile

import model_store
from face_pipeline import embed

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


async def enroll(name: str, files: List[UploadFile]) -> dict:
    """
    Process uploaded images for *name*, update the gallery in-place, and
    persist changes to disk.

    Returns a summary dict: {name, saved, failed, label, gallery_size}.
    """
    from config import IMAGE_SIZE, RAW_FACES_DIR
    from face_alignment import extract_face

    person_dir = ROOT / RAW_FACES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    saved: int = 0
    failed: list = []
    new_embeddings: list = []

    for upload in files:
        raw = await upload.read()
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            failed.append(upload.filename)
            continue

        result = extract_face(img)
        if result is None:
            failed.append(f"{upload.filename} (no face)")
            continue

        face_bgr, _, __ = result
        gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
        np.save(str(person_dir / f"{uuid.uuid4().hex}.npy"), resized)

        try:
            new_embeddings.append(embed(face_bgr))
        except Exception:
            log.exception("Embedding failed for %s", upload.filename)

        saved += 1

    # ── Update label map ──────────────────────────────────────────────────────
    lmap    = model_store.get_label_map()          # {int → name}
    inv_map = {v: k for k, v in lmap.items()}      # {name → int}

    if name in inv_map:
        new_label = inv_map[name]
    else:
        new_label = max(lmap.keys(), default=-1) + 1
        lmap[new_label] = name
        model_store.save_label_map(lmap)

    # ── Merge embeddings into gallery ─────────────────────────────────────────
    if new_embeddings:
        gallery = model_store.get_gallery()
        existing = gallery.get(new_label, None)

        if existing is not None:
            all_embs = ([existing] if isinstance(existing, np.ndarray) else list(existing)) + new_embeddings
        else:
            all_embs = new_embeddings

        gallery[new_label] = np.mean(np.array(all_embs), axis=0)
        model_store.save_gallery(gallery)

    return {
        "name":         name,
        "saved":        saved,
        "failed":       failed,
        "label":        new_label,
        "gallery_size": len(model_store.get_gallery()),
    }
