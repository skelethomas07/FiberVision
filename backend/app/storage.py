from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .config import Settings


class ObjectStorage(Protocol):
    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def get_to_path(self, key: str, destination: Path) -> None: ...
    def put_file(self, key: str, source: Path, content_type: str | None = None) -> None: ...


class LocalObjectStorage:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ValueError("invalid storage key")
        return path

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def get_to_path(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.get_bytes(key))

    def put_file(self, key: str, source: Path, content_type: str | None = None) -> None:
        self.put_bytes(key, source.read_bytes(), content_type)


class S3ObjectStorage:
    def __init__(self, settings: Settings):
        import boto3
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
        )
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        kwargs = {"Bucket": self.bucket, "Key": key, "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        self.client.put_object(**kwargs)

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def get_to_path(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(destination))

    def put_file(self, key: str, source: Path, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else None
        self.client.upload_file(str(source), self.bucket, key, ExtraArgs=extra or {})


def build_storage(settings: Settings) -> ObjectStorage:
    if settings.storage_backend.lower() == "s3":
        return S3ObjectStorage(settings)
    return LocalObjectStorage(settings.local_storage_dir)
