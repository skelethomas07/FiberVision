from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status

from ..models import ImageAsset
from ..services.scale_calibration import resolve_nm_per_pixel
from ..services.visionflux_import import (
    import_storage_key,
    inspect_sem_upload,
    measurement_payload,
    preview_storage_key,
)

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

    try:
        calibration = resolve_nm_per_pixel(data, nm_per_pixel)
        inspection = inspect_sem_upload(data, filename, calibration.nm_per_pixel)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"cannot read SEM image: {exc}") from exc

    with request.app.state.Session() as session:
        image = ImageAsset(
            original_filename=filename,
            content_type=file.content_type or "application/octet-stream",
            storage_key=key,
            size_bytes=len(data),
            nm_per_pixel=calibration.nm_per_pixel,
        )
        session.add(image)
        session.commit()
        session.refresh(image)

        request.app.state.storage.put_bytes(
            preview_storage_key(image.id), inspection.preview_bytes, inspection.preview_content_type
        )
        if inspection.is_visionflux_annotated:
            payload = {
                "kind": "visionflux_annotated",
                "measurements": [measurement_payload(m) for m in inspection.measurements],
            }
            request.app.state.storage.put_bytes(
                import_storage_key(image.id),
                json.dumps(payload).encode("utf-8"),
                "application/json",
            )

        return {
            "id": image.id,
            "filename": image.original_filename,
            "content_type": image.content_type,
            "size_bytes": image.size_bytes,
            "nm_per_pixel": image.nm_per_pixel,
            "calibration_source": calibration.source,
            "scale_label": calibration.scale_label,
            "scale_bar_px": calibration.scale_bar_px,
            "content_url": f"/api/images/{image.id}/content",
            "input_mode": "visionflux_annotated" if inspection.is_visionflux_annotated else "raw_sem",
            "imported_measurements": len(inspection.measurements),
        }


@router.get("/{image_id}/content")
def image_content(image_id: str, request: Request):
    with request.app.state.Session() as session:
        image = session.get(ImageAsset, image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="image not found")
        try:
            data = request.app.state.storage.get_bytes(preview_storage_key(image.id))
            return Response(content=data, media_type="image/png")
        except Exception:
            data = request.app.state.storage.get_bytes(image.storage_key)
            return Response(content=data, media_type=image.content_type)
