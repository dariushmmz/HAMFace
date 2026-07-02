"""model_store.py — process-wide lazy cache for model, gallery, and label map.

Import ``get_model``, ``get_gallery``, ``get_label_map`` anywhere; the heavy
objects are loaded once on first call and reused for the lifetime of the process.
Call ``reload_gallery()`` / ``reload_label_map()`` after gallery updates (e.g.
after enrolling a new person via the API).
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────
_model     = None
_loss_fn   = None
_gallery: Optional[Dict] = None
_label_map: Optional[Dict[int, str]] = None   # {int_label → person_name}


# ── Model ─────────────────────────────────────────────────────────────────────

def get_model():
    """Return the loaded HAMFace model, building it on first call."""
    global _model, _loss_fn
    if _model is not None:
        return _model

    from config import N_CLASSES, MODEL_WEIGHTS_PATH, CLASS_WEIGHTS_PATH
    from models import HAMFaceLoss, build_model, load_model

    root = Path(__file__).parent
    print(root)

    log.info("Loading HAMFace model from '%s' …", MODEL_WEIGHTS_PATH)
    # _model = build_model(N_CLASSES)
    # _model.load_weights(str(root / MODEL_WEIGHTS_PATH))

    _model, _loss_fn = load_model(n_classes=N_CLASSES, weights_path=str(root / MODEL_WEIGHTS_PATH),
                           class_weights_path=str(root / CLASS_WEIGHTS_PATH))
    _model.eval()

    log.info("Model loaded. %d classes.", N_CLASSES)
    return _model


# ── Gallery ───────────────────────────────────────────────────────────────────

def get_gallery() -> Dict:
    """Return the average-embedding gallery dict, loading from disk on first call."""
    global _gallery
    if _gallery is not None:
        return _gallery

    from config import GALLERY_AVG_PKL
    pkl_path = Path(__file__).parent / GALLERY_AVG_PKL

    if not pkl_path.exists():
        log.warning("Gallery not found at '%s'. Returning empty dict.", pkl_path)
        _gallery = {}
        return _gallery

    with open(pkl_path, "rb") as f:
        _gallery = pickle.load(f)

    log.info("Gallery loaded: %d enrolled persons.", len(_gallery))
    return _gallery


def reload_gallery() -> None:
    """Force the next ``get_gallery()`` call to re-read from disk."""
    global _gallery
    _gallery = None
    log.info("Gallery cache cleared — will reload on next request.")


# ── Label map ─────────────────────────────────────────────────────────────────

def get_label_map() -> Dict[int, str]:
    """Return the ``{integer_label → person_name}`` mapping."""
    global _label_map
    if _label_map is not None:
        return _label_map

    from config import DATASET_ROOT
    map_path = Path(__file__).parent / Path(DATASET_ROOT) / "label_map.npy"

    if not map_path.exists():
        log.warning("label_map.npy not found at '%s'.", map_path)
        _label_map = {}
        return _label_map

    raw        = np.load(str(map_path), allow_pickle=True).item()  # name → int
    _label_map = {v: k for k, v in raw.items()}                    # int  → name
    log.info("Label map loaded: %d persons.", len(_label_map))
    return _label_map


def reload_label_map() -> None:
    """Force the next ``get_label_map()`` call to re-read from disk."""
    global _label_map
    _label_map = None
    log.info("Label map cache cleared — will reload on next request.")