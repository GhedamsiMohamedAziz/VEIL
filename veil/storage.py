"""Content-addressed artifact storage.

Files land under `<artifact_root>/<org_id>/<sha256[:2]>/<sha256>`. The org id
is in the path so a directory listing can never leak across tenants, and the
digest means re-uploading the same file is free.

piggy: local filesystem only. The `uri` column already carries a scheme, so
an S3 backend can be added behind `put`/`open` without a data migration
(ADR-004).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from veil.config import get_settings

_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
}


def sniff_media_type(head: bytes) -> str | None:
    """Trust the bytes, not the client-supplied Content-Type."""
    for magic, mime in _MAGIC.items():
        if head.startswith(magic):
            return mime
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":  # bare RIFF is also WAV/AVI
        return "image/webp"
    # ISO base media; the brands that are MP4/QuickTime video, not JPEG 2000 or HEIF stills.
    if head[4:8] == b"ftyp" and head[8:12] in (
            b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1",
            b"mp4v", b"M4V ", b"qt  ", b"dash", b"mmp4"):
        return "video/mp4"
    return None


def _root() -> Path:
    return Path(get_settings().artifact_root)


def path_for(organization_id: str, digest: str) -> Path:
    return _root() / organization_id / digest[:2] / digest


def put_bytes(organization_id: str, data: bytes) -> tuple[str, str, int]:
    """Store bytes. Returns (uri, sha256, size)."""
    digest = hashlib.sha256(data).hexdigest()
    dest = path_for(organization_id, digest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        # Unique per writer: two uploads of the same bytes share `dest`, and a
        # shared temp name let one promote the other's half-written file.
        tmp = dest.with_suffix(f".{uuid.uuid4().hex}.part")
        tmp.write_bytes(data)
        tmp.replace(dest)
    return f"file://{dest.resolve()}", digest, len(data)


def put_stream(organization_id: str, stream: BinaryIO, max_bytes: int) -> tuple[str, str, int]:
    """Stream to disk with a hard size ceiling, hashing as we go."""
    tmp = _root() / organization_id / f"upload-{id(stream):x}.part"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    size = 0
    try:
        with tmp.open("wb") as fh:
            while chunk := stream.read(1024 * 256):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"upload exceeds {max_bytes} bytes")
                hasher.update(chunk)
                fh.write(chunk)
        digest = hasher.hexdigest()
        dest = path_for(organization_id, digest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            tmp.unlink()
        else:
            shutil.move(str(tmp), str(dest))
        return f"file://{dest.resolve()}", digest, size
    finally:
        tmp.unlink(missing_ok=True)


def local_path(uri: str) -> Path:
    if not uri.startswith("file://"):
        raise ValueError(f"unsupported artifact scheme: {uri.split('://')[0]}")
    path = Path(uri[len("file://") :])
    # Every stored blob is content-addressed: .../<sha[:2]>/<sha256>. Checking the
    # shape, not the root, keeps this independent of the cwd the API started in
    # while still refusing a URI that points at anything else on the disk.
    if not re.fullmatch(r"[0-9a-f]{64}", path.name) or path.parent.name != path.name[:2]:
        raise ValueError("not an artifact path")
    return path


def read_bytes(uri: str) -> bytes:
    return local_path(uri).read_bytes()
