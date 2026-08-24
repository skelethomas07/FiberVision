from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status

from ..models import ImageAsset

router = APIRouter(prefix="/api/images", tags=["images"])
_ALLOWED = {"image/jpeg", "image/png", "image/tiff", "image/bmp"}


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    nm_per_pixel: float | None = Form(default=None),
):
    if file.content_type not in _ALLOWED:
        raise HTTPException(status_code=415, detail="unsupported SEM image type")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="empty image")
    filename = Path(file.filename or "sem-image").name
    key = f"images/{uuid.uuid4()}/{filename}"
    request.app.state.storage.put_bytes(key, data, file.content_type)
    with request.app.state.Session() as session:
        image = ImageAsset(
            original_filename=filename,
            content_type=file.content_type or "application/octet-stream",
            storage_key=key,
            size_bytes=len(data),
            nm_per_pixel=nm_per_pixel,
        )
        session.add(image)
        session.commit()
        session.refresh(image)
        return {
            "id": image.id,
            "filename": image.original_filename,
            "content_type": image.content_type,
            "size_bytes": image.size_bytes,
            "nm_per_pixel": image.nm_per_pixel,
            "content_url": f"/api/images/{image.id}/content",
        }


@router.get("/{image_id}/content")
def image_content(image_id: str, request: Request):
    with request.app.state.Session() as session:
        image = session.get(ImageAsset, image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="image not found")
        data = request.app.state.storage.get_bytes(image.storage_key)
        return Response(content=data, media_type=image.content_type)
