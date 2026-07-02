"""face_alignment.py — MediaPipe Tasks API (≥ 0.10) face detection, alignment, and cropping.

Requires the face_landmarker.task model bundle. Download once with:
    wget -q -O face_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import math
import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple

# Suppress MediaPipe/glog "Feedback manager" and xnnpack delegate warnings
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp
import numpy as np

log = logging.getLogger(__name__)

# ── Tasks API imports ─────────────────────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
FaceLandmarker     = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

# Path to the downloaded .task model bundle
_MODEL_PATH = Path(__file__).parent / "checkpoints/face_landmarker.task"

# ── Landmark index constants ──────────────────────────────────────────────────
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172,  58, 132,  93, 234, 127, 162,  21,
     54, 103,  67, 109,
]
LEFT_EYE_INDICES  = [33, 133]
RIGHT_EYE_INDICES = [362, 263]

# Full eye contour indices for polygon drawing
LEFT_EYE_CONTOUR_INDICES = [
    33, 246, 161, 160, 159, 158, 157, 173,
    133, 155, 154, 153, 145, 144, 163, 7,
]
RIGHT_EYE_CONTOUR_INDICES = [
    362, 398, 384, 385, 386, 387, 388, 466,
    263, 249, 390, 373, 374, 380, 381, 382,
]

LIPS_INDICES = list({
    61, 146, 91, 181,  84,  17, 314, 405, 321, 375, 291, 308,
    78,  95, 88, 178,  87,  14, 317, 402, 318, 324,
})


# ── Landmarker factory ────────────────────────────────────────────────────────

def _make_landmarker() -> FaceLandmarker:
    """Create a single-image FaceLandmarker from the bundled .task model."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found at '{_MODEL_PATH}'.\n"
            "Download it with:\n"
            "  wget -q -O face_landmarker.task \\\n"
            "    https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
    options = FaceLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


# ── Module-level singleton ────────────────────────────────────────────────────
# Constructing a FaceLandmarker is expensive (~400-500 ms): it loads the .task
# model, initialises the xnnpack delegate, and spins up internal threads.
# Calling _make_landmarker() on every frame/image is the primary bottleneck.
# _get_landmarker() creates the instance once and reuses it for the lifetime of
# the process.  process_video() keeps its own context-managed instance because
# it already does so correctly; only extract_face() needed the fix.

_landmarker_instance: "FaceLandmarker | None" = None


def _get_landmarker() -> FaceLandmarker:
    """Return the process-wide FaceLandmarker, creating it on first call."""
    global _landmarker_instance
    if _landmarker_instance is None:
        _landmarker_instance = _make_landmarker()
        log.debug("[face_alignment] FaceLandmarker initialised (singleton).")
    return _landmarker_instance


# ── Internal helpers ──────────────────────────────────────────────────────────

def _bgr_to_mp_image(bgr: np.ndarray) -> mp.Image:
    """Convert a BGR numpy array to an SRGB mediapipe.Image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _detect_landmarks(bgr: np.ndarray, landmarker: FaceLandmarker):
    """
    Run face landmarking on *bgr*.

    Returns the first face's landmark list, or None if no face is found.
    The returned object is a list of NormalizedLandmark (access via [i].x/y/z).
    """
    result = landmarker.detect(_bgr_to_mp_image(bgr))
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]          # list[NormalizedLandmark]


def _get_coords(
    landmarks,
    indices: list[int],
    w: int, h: int,
) -> Tuple[list[int], list[int]]:
    xs = [int(landmarks[i].x * w) for i in indices]
    ys = [int(landmarks[i].y * h) for i in indices]
    return xs, ys


def _get_eye_centers(
    landmarks, h: int, w: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    def avg(indices):
        xs = [landmarks[i].x * w for i in indices]
        ys = [landmarks[i].y * h for i in indices]
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
    return avg(LEFT_EYE_INDICES), avg(RIGHT_EYE_INDICES)


def _align_face(bgr: np.ndarray, left_eye: tuple, right_eye: tuple) -> np.ndarray:
    dx, dy  = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle   = math.degrees(math.atan2(dy, dx))
    center  = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M       = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]))


def _extract_polygon_points(landmarks, h: int, w: int) -> np.ndarray:
    pts = [
        [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
        for idx in FACE_OVAL_INDICES
    ]
    return np.array(pts, dtype=np.int32)


def _extract_eye_polygons(
    landmarks, h: int, w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (left_eye_polygon, right_eye_polygon) as int32 arrays."""
    def pts(indices):
        return np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
            dtype=np.int32,
        )
    return pts(LEFT_EYE_CONTOUR_INDICES), pts(RIGHT_EYE_CONTOUR_INDICES)


def _crop_face_with_polygon(
    bgr: np.ndarray, polygon: np.ndarray
) -> Optional[np.ndarray]:
    mask   = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(bgr, bgr, mask=mask)

    x, y, bw, bh = cv2.boundingRect(polygon)
    x1, y1 = max(0, x),                  max(0, y)
    x2, y2 = min(bgr.shape[1], x + bw),  min(bgr.shape[0], y + bh)

    crop = masked[y1:y2, x1:x2]
    if crop.size == 0:
        log.warning("Empty face crop detected.")
        return None
    return crop


# ── Annotation helpers ────────────────────────────────────────────────────────

def _draw_face_overlay(
    frame: np.ndarray,
    face_polygon: np.ndarray,
    alpha: float = 0.20,
) -> np.ndarray:
    """
    Blend a white semi-transparent fill inside *face_polygon* onto *frame*.

    Parameters
    ----------
    frame        : BGR frame to annotate (modified in place and returned).
    face_polygon : Nx2 int32 polygon points.
    alpha        : Opacity of the white fill (0 = invisible, 1 = solid white).
    """
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, face_polygon, (255, 255, 255))
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return frame


def _draw_eye_polygons(
    frame: np.ndarray,
    left_eye_pts: np.ndarray,
    right_eye_pts: np.ndarray,
    color: Tuple[int, int, int] = (200, 220, 255),
    fill_alpha: float = 0.40,
    line_thickness: int = 1,
) -> np.ndarray:
    """
    Draw filled + outlined polygons for both eyes on *frame*.

    Parameters
    ----------
    frame           : BGR frame (modified in place).
    left_eye_pts    : Contour points for the left eye.
    right_eye_pts   : Contour points for the right eye.
    color           : BGR colour for the polygon fill and outline.
    fill_alpha      : Opacity of the fill layer.
    line_thickness  : Outline stroke width in pixels.
    """
    overlay = frame.copy()
    for pts in (left_eye_pts, right_eye_pts):
        cv2.fillConvexPoly(overlay, pts, color)
    cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
    for pts in (left_eye_pts, right_eye_pts):
        cv2.polylines(frame, [pts], isClosed=True, color=color,
                      thickness=line_thickness, lineType=cv2.LINE_AA)
    return frame


class FPSTracker:
    """Rolling-window FPS tracker."""

    def __init__(self, window: int = 30):
        self._window = window
        self._times: list[float] = []

    def tick(self) -> float:
        """Record the current timestamp and return the smoothed FPS."""
        now = time.monotonic()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])

    def draw(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Overlay the FPS value in the top-right corner of *frame*."""
        text  = f"FPS: {fps:.1f}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thick = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x = frame.shape[1] - tw - 12
        y = th + 10
        # Dark shadow for legibility over any background
        cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0),    thick + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x,     y),     font, scale, (255, 255, 255), thick,     cv2.LINE_AA)
        return frame


# ── Occlusion helpers (used during dataset construction) ──────────────────────

def occlude_eyes(
    image: np.ndarray, landmarks, h: int, w: int, padding: int = 10
) -> np.ndarray:
    """Draw a black rectangle over both eyes (in-place on a copy)."""
    lxs, lys = _get_coords(landmarks, LEFT_EYE_INDICES,  w, h)
    rxs, rys = _get_coords(landmarks, RIGHT_EYE_INDICES, w, h)
    all_x, all_y = lxs + rxs, lys + rys
    cv2.rectangle(
        image,
        (max(0, min(all_x) - padding - 5),  max(0, min(all_y) - padding - 12)),
        (min(w, max(all_x) + padding + 5),  min(h, max(all_y) + padding + 12)),
        (0, 0, 0), thickness=-1,
    )
    return image


def occlude_lips(
    image: np.ndarray, landmarks, h: int, w: int, padding: int = 10
) -> np.ndarray:
    """Draw a black rectangle over the lip region (in-place on a copy)."""
    xs, ys = _get_coords(landmarks, LIPS_INDICES, w, h)
    cv2.rectangle(
        image,
        (max(0, min(xs) - padding - 25), max(0, min(ys) - padding - 5)),
        (min(w, max(xs) + padding + 25), min(h, max(ys) + padding)),
        (0, 0, 0), thickness=-1,
    )
    return image


# ── Public API ────────────────────────────────────────────────────────────────
#
# Pipeline order (both functions below): YOLO locates face box(es) → each box
# is cropped out of the original image → MediaPipe FaceLandmarker runs only
# on that crop (detect → eye-centre align → re-detect → oval/eye polygons).
# MediaPipe is never run against the full, un-cropped frame anymore — YOLO is
# the localiser, MediaPipe is the aligner/landmarker.

def _align_and_extract(
    crop_bgr: np.ndarray,
    landmarker: FaceLandmarker,
    with_occlusions: bool = True,
):
    """
    Run the detect → align → re-detect → polygon-crop steps on an
    already-localised face crop (typically a YOLO box crop).

    Returns ``(face_crop, eyes_occluded, lips_occluded)`` — the latter two
    are ``None`` when *with_occlusions* is False — or ``None`` if no face
    landmarks could be found in *crop_bgr*.
    """
    landmarks = _detect_landmarks(crop_bgr, landmarker)
    if landmarks is None:
        return None

    h, w = crop_bgr.shape[:2]
    left_eye, right_eye = _get_eye_centers(landmarks, h, w)
    aligned = _align_face(crop_bgr, left_eye, right_eye)

    # Re-detect on the rotated crop for accurate polygon / occlusion coords
    landmarks = _detect_landmarks(aligned, landmarker)
    if landmarks is None:
        return None

    h, w = aligned.shape[:2]
    polygon = _extract_polygon_points(landmarks, h, w)

    face_crop = _crop_face_with_polygon(aligned, polygon)
    if face_crop is None:
        return None

    if not with_occlusions:
        return face_crop, None, None

    eyes_occluded = _crop_face_with_polygon(
        occlude_eyes(aligned.copy(), landmarks, h, w), polygon
    )
    lips_occluded = _crop_face_with_polygon(
        occlude_lips(aligned.copy(), landmarks, h, w), polygon
    )

    if eyes_occluded is None or lips_occluded is None:
        return None

    return face_crop, eyes_occluded, lips_occluded


def extract_face(
    image: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Locate the dominant (largest) face in *image* (BGR) with YOLO, then
    align/crop it with MediaPipe.

    Used for single-face workflows such as enrollment, where each upload is
    expected to contain one person.

    Returns
    -------
    ``(face_crop, occluded_eyes_crop, occluded_lips_crop)``
    Three BGR crops of the same region, or ``None`` if no face is found.
    """
    from face_detector import detect_faces

    boxes = detect_faces(image)
    if not boxes:
        log.warning("[face_alignment] YOLO found no face in input image.")
        return None

    x1, y1, x2, y2 = boxes[0]   # largest box — boxes are pre-sorted by area
    box_crop = image[y1:y2, x1:x2]
    if box_crop.size == 0:
        return None

    landmarker = _get_landmarker()
    result = _align_and_extract(box_crop, landmarker, with_occlusions=True)
    if result is None:
        log.warning("[face_alignment] MediaPipe found no landmarks in YOLO crop.")
        return None
    return result


def extract_faces(
    image: np.ndarray,
) -> List[dict]:
    """
    Locate *all* faces in *image* (BGR) with YOLO, then align/crop each with
    MediaPipe. Used for multi-person workflows such as live recognition.

    Returns a list of dicts, one per successfully-detected face::

        [{
            "bbox":          (x1, y1, x2, y2),   # frame coords, padded YOLO box
            "face_crop":      np.ndarray,         # aligned crop, for embedding
            "polygon":        np.ndarray (N,2),   # face-oval pts, FRAME coords
            "left_eye_pts":   np.ndarray (M,2),   # left-eye contour, FRAME coords
            "right_eye_pts":  np.ndarray (M,2),   # right-eye contour, FRAME coords
        }, …]

    ``polygon``/``left_eye_pts``/``right_eye_pts`` come from the landmarks
    detected directly on the (unrotated) YOLO box crop, offset back into the
    full frame's coordinate space — so they can be drawn straight onto the
    original frame. ``face_crop`` is produced separately via eye-centre
    rotation + re-detection (more accurate for embedding, but its own
    coordinate space, so it isn't used for the overlay).

    Boxes where MediaPipe couldn't find landmarks (e.g. extreme profile, too
    small, blurred) are silently skipped. Empty list if YOLO finds nothing.
    """
    from face_detector import detect_faces

    boxes = detect_faces(image)
    if not boxes:
        return []

    landmarker = _get_landmarker()
    out: List[dict] = []

    for (x1, y1, x2, y2) in boxes:
        box_crop = image[y1:y2, x1:x2]
        if box_crop.size == 0:
            continue

        # ── Raw (pre-alignment) landmarks → overlay polygons in frame coords ──
        raw_landmarks = _detect_landmarks(box_crop, landmarker)
        if raw_landmarks is None:
            continue

        bh, bw   = box_crop.shape[:2]
        offset   = np.array([x1, y1], dtype=np.int32)
        polygon       = (_extract_polygon_points(raw_landmarks, bh, bw) + offset).astype(np.int32)
        left_eye_pts, right_eye_pts = _extract_eye_polygons(raw_landmarks, bh, bw)
        left_eye_pts  = (left_eye_pts  + offset).astype(np.int32)
        right_eye_pts = (right_eye_pts + offset).astype(np.int32)

        # ── Aligned crop → used for embedding only ─────────────────────────────
        result = _align_and_extract(box_crop, landmarker, with_occlusions=False)
        if result is None:
            continue
        face_crop, _, __ = result

        out.append({
            "bbox":         (x1, y1, x2, y2),
            "face_crop":    face_crop,
            "polygon":      polygon,
            "left_eye_pts":  left_eye_pts,
            "right_eye_pts": right_eye_pts,
        })

    return out


# ── Convenience loader ────────────────────────────────────────────────────────

def load_aligned_grayscale(img_path: str, image_size: int = 128) -> Optional[np.ndarray]:
    """
    Load *img_path*, align + crop the face, convert to grayscale, resize to image_size².

    Returns ``None`` if the file can't be read, is all-black, or has no face.
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        log.warning("Could not read: %s", img_path)
        return None
    if bgr.max() == 0:
        log.warning("All-black image, skipping: %s", img_path)
        return None

    result = extract_face(bgr)
    if result is None:
        return None

    face_bgr, _, __ = result
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (image_size, image_size))


# ── Video reader ──────────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    image_size: int = 128,
    frame_skip: int = 1,
    save_path: Optional[str] = None,
    show: bool = True,
) -> list[np.ndarray]:
    """
    Read every *frame_skip*-th frame from *video_path*, extract and align the
    dominant face in each frame, and optionally display / save the results.

    Parameters
    ----------
    video_path  : Path to the input video file.
    image_size  : Side length (px) of the square grayscale face crop saved to
                  the returned list.  The annotated preview window keeps the
                  original frame resolution.
    frame_skip  : Process 1 in every *frame_skip* frames (1 = every frame).
    save_path   : Optional path for the annotated output video (MP4).
    show        : Whether to display a live preview window (press Q or Esc to
                  stop early).

    Returns
    -------
    A list of grayscale face-crop arrays (shape ``image_size × image_size``),
    one per successfully processed frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"[face_alignment] Cannot open video: {video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_src, (width, height))
        print(f"[face_alignment] Saving annotated video to: {save_path}")

    face_crops: list[np.ndarray] = []
    frame_idx   = 0
    fps_tracker = FPSTracker(window=30)

    with _make_landmarker() as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()

            if frame_idx % frame_skip == 0:
                landmarks = _detect_landmarks(frame, landmarker)

                if landmarks is not None:
                    h, w = frame.shape[:2]
                    left_eye, right_eye = _get_eye_centers(landmarks, h, w)
                    aligned = _align_face(frame, left_eye, right_eye)

                    lm2 = _detect_landmarks(aligned, landmarker)
                    if lm2 is not None:
                        h2, w2  = aligned.shape[:2]
                        polygon = _extract_polygon_points(lm2, h2, w2)
                        left_eye_pts, right_eye_pts = _extract_eye_polygons(lm2, h2, w2)
                        face_crop = _crop_face_with_polygon(aligned, polygon)

                        if face_crop is not None:
                            # Grayscale crop → returned list
                            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                            face_crops.append(cv2.resize(gray, (image_size, image_size)))

                            # 1. White semi-transparent fill over the face region
                            _draw_face_overlay(annotated, polygon, alpha=0.20)

                            # 2. Green face-oval outline
                            cv2.polylines(
                                annotated, [polygon],
                                isClosed=True, color=(255, 255, 255),
                                thickness=2, lineType=cv2.LINE_AA,
                            )

                            # 3. Light-blue filled + outlined eye polygons
                            _draw_eye_polygons(
                                annotated, left_eye_pts, right_eye_pts,
                                color=(255, 255, 255), fill_alpha=0.40,
                            )

                # Status bar — bottom-left
                status = f"Frame {frame_idx}/{total}  Crops: {len(face_crops)}"
                cv2.putText(
                    annotated, status, (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1, cv2.LINE_AA,
                )

            # FPS counter — top-right, ticked every frame for accuracy
            fps_live = fps_tracker.tick()
            fps_tracker.draw(annotated, fps_live)

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("face_alignment — video", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[face_alignment] User stopped early.")
                    break

            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"[face_alignment] Done. Saved: {save_path}")
    if show:
        cv2.destroyAllWindows()

    print(f"[face_alignment] Extracted {len(face_crops)} face crops from {frame_idx} frames.")
    return face_crops


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    IMAGE_SIZE = 128  # spatial resolution fed to the model

    parser = argparse.ArgumentParser(
        description="face_alignment.py — run face alignment on an image or video."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=str, help="Path to a single image file.")
    source.add_argument("--video", type=str, help="Path to a video file.")

    parser.add_argument(
        "--size", type=int, default=IMAGE_SIZE,
        help=f"Output face-crop size in pixels (default: {IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=1,
        help="Process 1 in every N frames for video input (default: 1).",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save annotated output video to this path (video mode only).",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Suppress the preview window.",
    )
    args = parser.parse_args()

    # ── Image mode ────────────────────────────────────────────────────────────
    if args.image:
        face = load_aligned_grayscale(args.image, image_size=args.size)
        if face is None:
            print("[__main__] No face found or image unreadable.")
        else:
            print(f"[__main__] Face crop shape: {face.shape}")
            if not args.no_display:
                cv2.imshow("Aligned face", face)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # ── Video mode ────────────────────────────────────────────────────────────
    elif args.video:
        crops = process_video(
            video_path=args.video,
            image_size=args.size,
            frame_skip=args.frame_skip,
            save_path=args.save,
            show=not args.no_display,
        )
        print(f"[__main__] Total face crops collected: {len(crops)}")