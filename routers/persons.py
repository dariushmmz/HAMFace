"""routers/persons.py — person listing and system status endpoints."""

from fastapi import APIRouter

import model_store

router = APIRouter(tags=["persons"])


@router.get("/api/status")
async def status():
    gallery = model_store.get_gallery()
    lmap    = model_store.get_label_map()
    return {
        "gallery_size":   len(gallery),
        "known_persons":  list(lmap.values()),
        "model_ready":    model_store._model is not None,
    }


@router.get("/api/persons")
async def list_persons():
    lmap    = model_store.get_label_map()
    gallery = model_store.get_gallery()
    return {
        "persons": [
            {"id": k, "name": v, "in_gallery": k in gallery}
            for k, v in lmap.items()
        ]
    }
