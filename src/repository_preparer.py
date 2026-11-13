"""Materialize explicitly supplied local repository archives without network access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from typing import Any


class RepositoryPreparationError(RuntimeError):
    """Raised when a local repository archive cannot be prepared safely."""


def _safe_member_path(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    if not target.is_relative_to(root.resolve()):
        raise RepositoryPreparationError(f"Archive member escapes extraction root: {member}")
    return target


def materialize_archive(archive: Path, output_root: Path) -> dict[str, Any]:
    if not archive.is_file():
        return {
            "status": "BLOCKED",
            "category": "REPOSITORY_UNAVAILABLE",
            "message": str(archive),
        }
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = _safe_member_path(output_root, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_stream, target.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)
    return {
        "status": "READY",
        "category": "ARCHIVE_MATERIALIZED",
        "archive": str(archive.resolve()),
        "archive_sha256": digest,
        "root": str(output_root.resolve()),
    }


def write_preparation_record(path: Path, records: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
