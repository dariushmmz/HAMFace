"""face_pipeline.py — embedding, matching, annotation, and frame processing.

All MediaPipe calls go through face_alignment (Tasks API ≥ 0.10).
This module is the single place that combines alignment → embedding → matching
and is used by both the CLI (main.py) and the FastAPI dashboard (dashboard/app.py).
"""

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Lazy imports to avoid circular deps and heavy load at startup ─────────────

def _model_store():
    """Return (get_model, get_gallery, get_label_map) from model_store."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from model_store import get_gallery, get_label_map, get_model
    return get_model, get_gallery, get_label_map


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_face(face_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert a BGR face crop to a model-ready float32 tensor.

    Mirrors the preprocessing in ``inference.load_image`` used by main.py:
    grayscale → resize → replicate to 3-channel → ``(1, 3, H, W)`` NCHW tensor
    on the active device.

    Returns shape ``(1, 3, IMAGE_SIZE, IMAGE_SIZE)``.
    """
    from config import IMAGE_SIZE
    gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    # (H, W) → (H, W, 1) → (H, W, 3) float32
    arr = resized[:, :, np.newaxis].astype(np.float32)
    arr = np.concatenate([arr] * 3, axis=-1)
    # (H, W, C) → (C, H, W) → (1, C, H, W)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(_device)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(face_bgr: np.ndarray) -> np.ndarray:
    """
    Return an L2-normalised embedding vector for *face_bgr*.

    Uses the PyTorch HAMFace model via ``model_store.get_model()``, consistent
    with the ``_embed`` helper in main.py.
    """
    get_model, _, __ = _model_store()
    model = get_model()
    data  = preprocess_face(face_bgr)           # (1, 3, H, W) on _device
    with torch.no_grad():
        emb = model(data, data, training=False)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()[0]                 # (embed_dim,)


# ── Matching ──────────────────────────────────────────────────────────────────

def match(embedding: np.ndarray, threshold: float = 0.45):
    """
    Compare *embedding* against the in-memory gallery using cosine similarity.

    Parameters
    ----------
    embedding:
        Shape ``(embed_dim,)`` — must be L2-normalised.
    threshold:
        Minimum score to accept as a known identity.

    Returns
    -------
    ``(name, score)`` — name is ``"unknown"`` when score < threshold.
    """
    _, get_gallery, get_label_map = _model_store()
    gallery = get_gallery()
    lmap    = get_label_map()

    if not gallery:
        return "unknown", 0.0

    query = embedding.reshape(1, -1)            # (1, embed_dim)
    best_label = max(
        gallery,
        key=lambda lbl: cosine_similarity(query, gallery[lbl].reshape(1, -1))[0][0],
    )
    best_score = float(
        cosine_similarity(query, gallery[best_label].reshape(1, -1))[0][0]
    )

    name = lmap.get(best_label, f"ID:{best_label}")
    if best_score < threshold:
        name = "unknown"
    return name, best_score


# ── Annotation ────────────────────────────────────────────────────────────────

def _text_scale(frame: np.ndarray) -> tuple[float, int]:
    """
    Derive a font scale and thickness proportional to the shorter image side.

    A 640-px-wide frame gets scale≈0.55/thickness=1; a 1920-px frame gets
    scale≈1.65/thickness=2.  Clamps to [0.35, 2.5].
    """
    short_side = min(frame.shape[:2])           # height or width, whichever is smaller
    scale      = max(0.35, min(2.5, short_side / 1000.0))
    thickness  = 2 if scale >= 1.0 else 1
    return scale, thickness


def draw_face_overlay(
    frame: np.ndarray,
    polygon_pts: np.ndarray,
    left_eye_pts: np.ndarray,
    right_eye_pts: np.ndarray,
    name: str,
    score: float,
    face_alpha: float = 0.18,
    eye_alpha: float = 0.40,
) -> np.ndarray:
    """
    Draw a white semi-transparent face-oval fill, eye-contour fills, white
    outlines, and a name/score label on *frame* in-place.

    Parameters
    ----------
    frame:
        BGR image to annotate.
    polygon_pts:
        Shape ``(N, 2)`` int32 array — face-oval contour.
    left_eye_pts / right_eye_pts:
        Shape ``(M, 2)`` int32 arrays — eye contour polygons from
        ``face_alignment._extract_eye_polygons``.
    name / score:
        Identity and confidence to display.
    face_alpha:
        Opacity of the white face-oval fill (0 = invisible, 1 = solid white).
    eye_alpha:
        Opacity of the white eye fills.

    Returns
    -------
    The same *frame* (annotated in-place).
    """
    if polygon_pts is None or len(polygon_pts) == 0:
        return frame

    WHITE = (255, 255, 255)

    # ── 1. Semi-transparent white fill over face oval ─────────────────────────
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, polygon_pts, WHITE)
    cv2.addWeighted(overlay, face_alpha, frame, 1.0 - face_alpha, 0, frame)

    # ── 2. White face-oval outline ────────────────────────────────────────────
    cv2.polylines(frame, [polygon_pts.reshape((-1, 1, 2))],
                  isClosed=True, color=WHITE, thickness=2, lineType=cv2.LINE_AA)

    # ── 3. Semi-transparent white eye fills + outlines ────────────────────────
    if left_eye_pts is not None and right_eye_pts is not None:
        eye_overlay = frame.copy()
        for eye_pts in (left_eye_pts, right_eye_pts):
            if eye_pts is not None and len(eye_pts):
                cv2.fillConvexPoly(eye_overlay, eye_pts, WHITE)
        cv2.addWeighted(eye_overlay, eye_alpha, frame, 1.0 - eye_alpha, 0, frame)
        for eye_pts in (left_eye_pts, right_eye_pts):
            if eye_pts is not None and len(eye_pts):
                cv2.polylines(frame, [eye_pts.reshape((-1, 1, 2))],
                              isClosed=True, color=WHITE, thickness=1,
                              lineType=cv2.LINE_AA)

    # ── 4. Label with dark semi-transparent background ────────────────────────
    scale, thick = _text_scale(frame)
    font  = cv2.FONT_HERSHEY_DUPLEX
    label = f"{name}  {score:.2f}"

    x = int(polygon_pts[:, 0].min())
    y = int(polygon_pts[:, 1].min())

    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
    pad    = max(4, int(scale * 6))
    bg_tl  = (x, y - th - pad * 2 - baseline)
    bg_br  = (x + tw + pad * 2, y)

    # Dark translucent pill background
    label_overlay = frame.copy()
    cv2.rectangle(label_overlay, bg_tl, bg_br, (20, 20, 20), -1)
    cv2.addWeighted(label_overlay, 0.60, frame, 0.40, 0, frame)

    # White text on top
    cv2.putText(frame, label, (x + pad, y - baseline - pad // 2),
                font, scale, WHITE, thick, cv2.LINE_AA)

    return frame


# ── Annotation: per-face bounding box + label ─────────────────────────────────
# Used by process_frame(). Unlike draw_face_overlay() (which needs the
# oval/eye polygons from a *single* full-frame MediaPipe pass), this draws
# straight off the YOLO box in the original frame's coordinate space — which
# is what we have once each face is detected/aligned independently in its
# own cropped + rotated coordinate system.

def draw_face_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    name: str,
    score: float,
) -> np.ndarray:
    """Draw a white bounding box and a name/score label above it on *frame* in-place."""
    x1, y1, x2, y2 = bbox
    WHITE = (255, 255, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, thickness=2, lineType=cv2.LINE_AA)

    scale, thick = _text_scale(frame)
    font  = cv2.FONT_HERSHEY_DUPLEX
    label = f"{name}  {score:.2f}"

    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
    pad   = max(4, int(scale * 6))
    bg_tl = (x1, max(0, y1 - th - pad * 2 - baseline))
    bg_br = (x1 + tw + pad * 2, y1)

    label_overlay = frame.copy()
    cv2.rectangle(label_overlay, bg_tl, bg_br, (20, 20, 20), -1)
    cv2.addWeighted(label_overlay, 0.60, frame, 0.40, 0, frame)

    cv2.putText(frame, label, (x1 + pad, y1 - baseline - pad // 2),
                font, scale, WHITE, thick, cv2.LINE_AA)

    return frame


# ── Full frame pipeline ───────────────────────────────────────────────────────

def process_frame(frame_bgr: np.ndarray, threshold: float = 0.45):
    """
    Run the complete recognition pipeline on one BGR frame.

    Steps
    -----
    1. YOLO locates every face box in the frame (``face_alignment.extract_faces``).
    2. Each box is cropped and handed to MediaPipe for eye-centre alignment
       and oval-polygon cropping.
    3. Each aligned face crop is embedded with the HAMFace model (PyTorch, NCHW).
    4. Each embedding is matched against the gallery.
    5. Every detected face is annotated with a bounding box + name/score label.

    Parameters
    ----------
    frame_bgr:
        Raw BGR frame (e.g. from ``cv2.imread`` or a webcam capture).
    threshold:
        Cosine-similarity threshold below which a face is labelled "unknown".

    Returns
    -------
    ``(annotated_frame, results)``

    *results* is a list of dicts::

        [{"name": str, "score": float, "bbox": [x1, y1, x2, y2]}, …]

    One entry per face YOLO found and MediaPipe could align. An empty list
    means no face was detected (or none could be aligned).
    """
    from face_alignment import extract_faces

    annotated = frame_bgr.copy()
    faces     = extract_faces(frame_bgr)

    if not faces:
        log.debug("No face detected in frame.")
        return annotated, []

    results = []
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        face_crop       = face["face_crop"]
        poly_pts        = face["polygon"]
        left_eye_pts    = face["left_eye_pts"]
        right_eye_pts   = face["right_eye_pts"]

        try:
            emb         = embed(face_crop)
            name, score = match(emb, threshold)
        except Exception:
            log.exception("Recognition failed on a detected face.")
            name, score = "error", 0.0

        draw_face_overlay(annotated, poly_pts, left_eye_pts, right_eye_pts, name, score)

        results.append({
            "name":  name,
            "score": round(score, 4),
            "bbox":  [x1, y1, x2, y2],
        })

    return annotated, results