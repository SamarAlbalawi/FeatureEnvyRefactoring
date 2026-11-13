"""Prepare reproducible, isolated source workspaces for proposal validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile


class ValidationWorkspaceError(RuntimeError):
    """Raised when an archive cannot be verified or isolated safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_commit_metadata(path: Path, expected_commit: str) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationWorkspaceError(
            f"Commit metadata is not valid JSON: {path}"
        ) from error
    if metadata.get("sha") != expected_commit:
        raise ValidationWorkspaceError(
            "Commit metadata does not match the requested commit."
        )
    return metadata


def prepare_workspace(
    *,
    archive_path: Path,
    commit_metadata_path: Path,
    expected_commit: str,
    output_root: Path,
) -> Path:
    """Verify an official pinned archive and extract it to a new directory."""
    if not archive_path.is_file():
        raise ValidationWorkspaceError(f"Archive does not exist: {archive_path}")
    commit_metadata = _load_commit_metadata(commit_metadata_path, expected_commit)
    expected_prefix = f"Newtonsoft.Json-{expected_commit}/"
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if not names or not all(name.startswith(expected_prefix) for name in names):
            raise ValidationWorkspaceError(
                "Archive root does not match the requested commit prefix."
            )
        required = {
            expected_prefix + "Src/Newtonsoft.Json/Newtonsoft.Json.csproj",
            expected_prefix
            + "Src/Newtonsoft.Json/Serialization/DefaultContractResolver.cs",
        }
        if not required.issubset(names):
            raise ValidationWorkspaceError("Archive is missing required source files.")

    output_root.mkdir(parents=True, exist_ok=True)
    workspace = output_root / f"newtonsoft-{expected_commit[:12]}"
    if workspace.exists():
        raise ValidationWorkspaceError(
            f"Refusing to overwrite existing validation workspace: {workspace}"
        )
    workspace.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not member.filename.startswith(expected_prefix):
                raise ValidationWorkspaceError(
                    f"Archive member is outside the expected root: {member.filename}"
                )
            relative = Path(member.filename[len(expected_prefix) :])
            if not relative.parts:
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise ValidationWorkspaceError(
                    f"Archive contains an unsafe path: {member.filename}"
                )
            destination = workspace / relative
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                target.write(source.read())
    manifest = {
        "repository": "JamesNK/Newtonsoft.Json",
        "commit": expected_commit,
        "commit_api_sha": commit_metadata["sha"],
        "commit_api_verified": commit_metadata.get("commit", {})
        .get("verification", {})
        .get("verified"),
        "archive": str(archive_path.resolve()),
        "archive_sha256": _sha256(archive_path),
        "workspace": str(workspace.resolve()),
        "source_root": str(workspace.resolve()),
        "patch_application": "not performed",
        "original_checkout_modified": False,
    }
    (workspace / "workspace-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--commit-metadata", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    workspace = prepare_workspace(
        archive_path=args.archive,
        commit_metadata_path=args.commit_metadata,
        expected_commit=args.commit,
        output_root=args.output_root,
    )
    print(f"VALIDATION WORKSPACE: {workspace.resolve()}")
    print("COMMIT VERIFIED: YES")
    print("PATCH APPLIED: NO")


if __name__ == "__main__":
    main()
