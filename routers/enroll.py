"""routers/enroll.py — /api/enroll/* endpoints."""

from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import person_store

router = APIRouter(prefix="/api/enroll", tags=["enroll"])


@router.post("/person")
async def enroll_person(
    name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    Add a new person to the dataset and rebuild the gallery.
    Saves uploaded images to dataset/raw_faces/{name}/ and updates
    gallery_avg.pkl in-place.
    """
    result = await person_store.enroll(name, files)
    if result["saved"] == 0:
        raise HTTPException(400, f"No faces extracted. Failed: {result['failed']}")
    return result
