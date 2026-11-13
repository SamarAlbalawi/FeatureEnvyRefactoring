"""Deterministically normalize and apply the constrained proposal patches."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class PatchNormalizationError(ValueError):
    """Raised when a proposal patch is malformed or ambiguous."""


def parse_changed_files(response: dict[str, Any]) -> list[dict[str, str]]:
    """Validate and extract the repository edits from a model response.

    The model schema nests the patch under ``proposal.changed_files[].edit``.
    This function performs schema validation only; it never rewrites patch text.
    """
    if not isinstance(response, dict):
        raise PatchNormalizationError("Response must be a JSON object.")
    proposal = response.get("proposal")
    if not isinstance(proposal, dict):
        raise PatchNormalizationError("Response has no proposal object.")
    changed_files = proposal.get("changed_files")
    if not isinstance(changed_files, list):
        raise PatchNormalizationError("Proposal changed_files must be a list.")

    parsed: list[dict[str, str]] = []
    for index, changed_file in enumerate(changed_files):
        if not isinstance(changed_file, dict):
            raise PatchNormalizationError(f"changed_files[{index}] must be an object.")
        path = changed_file.get("path")
        operation = changed_file.get("operation")
        edit = changed_file.get("edit")
        if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise PatchNormalizationError(f"changed_files[{index}] has an unsafe path.")
        if operation not in {"modify", "add", "delete"}:
            raise PatchNormalizationError(f"changed_files[{index}] has an unsupported operation.")
        if not isinstance(edit, dict) or edit.get("type") != "unified_diff":
            raise PatchNormalizationError(f"changed_files[{index}] has an invalid edit object.")
        content = edit.get("content")
        if not isinstance(content, str):
            raise PatchNormalizationError(f"changed_files[{index}] edit content must be text.")
        parsed.append({"path": path, "operation": operation, "content": content})
    return parsed


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$")


def normalize_unified_diff(raw_patch: str, repository_path: str) -> tuple[str, list[dict[str, int | str]]]:
    """Add headers and recompute hunk counts without changing hunk body lines."""
    lines = raw_patch.splitlines()
    if not lines or any(line.startswith(("--- ", "+++ ")) for line in lines):
        raise PatchNormalizationError("Expected a headerless unified diff.")
    if not repository_path or Path(repository_path).is_absolute() or ".." in Path(repository_path).parts:
        raise PatchNormalizationError("Repository path is not a safe relative path.")

    hunks: list[str] = []
    operations: list[dict[str, int | str]] = []
    index = 0
    while index < len(lines):
        match = _HUNK.match(lines[index])
        if match is None:
            raise PatchNormalizationError(f"Expected hunk header at line {index + 1}.")
        original_header = lines[index]
        body: list[str] = []
        index += 1
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if line.startswith("\\ No newline at end of file"):
                body.append(line)
            elif line.startswith((" ", "+", "-")):
                body.append(line)
            else:
                raise PatchNormalizationError(f"Malformed hunk body at line {index + 1}.")
            index += 1

        old_count = sum(line.startswith((" ", "-")) for line in body)
        new_count = sum(line.startswith((" ", "+")) for line in body)
        old_start = int(match.group(1))
        new_start = int(match.group(3))
        suffix = original_header[match.end() :]
        hunks.append(
            f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}\n"
            + "\n".join(body)
        )
        operations.append(
            {
                "operation": "recalculated_hunk_counts",
                "old_start": old_start,
                "new_start": new_start,
                "old_count": old_count,
                "new_count": new_count,
            }
        )

    normalized = f"--- a/{repository_path}\n+++ b/{repository_path}\n" + "\n".join(hunks) + "\n"
    operations.insert(0, {"operation": "added_file_headers", "path": repository_path})
    return normalized, operations


def apply_unified_diff(source: Path, normalized_patch: str) -> dict[str, int]:
    """Apply one normalized patch using exact line-start and context matching."""
    patch_lines = normalized_patch.splitlines()
    if len(patch_lines) < 3 or patch_lines[1] != patch_lines[0].replace("--- a/", "+++ b/"):
        raise PatchNormalizationError("Normalized file headers are invalid.")
    original = source.read_text(encoding="utf-8", newline="")
    source_lines = original.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in original else "\n"
    output: list[str] = []
    cursor = 0
    index = 2
    applied = 0
    while index < len(patch_lines):
        match = _HUNK.match(patch_lines[index])
        if match is None:
            raise PatchNormalizationError("Malformed normalized hunk header.")
        old_start = int(match.group(1))
        body: list[str] = []
        index += 1
        while index < len(patch_lines) and not patch_lines[index].startswith("@@ "):
            body.append(patch_lines[index])
            index += 1
        position = old_start - 1
        if position < cursor:
            raise PatchNormalizationError("Overlapping or reordered hunk.")
        output.extend(source_lines[cursor:position])
        check = position
        for line in body:
            if line.startswith("\\ No newline"):
                continue
            prefix, text = line[0], line[1:]
            if prefix in " -":
                if check >= len(source_lines) or source_lines[check].rstrip("\r\n") != text:
                    raise PatchNormalizationError(f"Source context mismatch at line {check + 1}.")
                check += 1
            if prefix in " +":
                output.append(text + newline)
        cursor = check
        applied += 1
    output.extend(source_lines[cursor:])
    source.write_text("".join(output), encoding="utf-8", newline="")
    return {"hunks_applied": applied}


def apply_changed_files(
    workspace_root: str | Path,
    changed_files: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Apply validated modify/add/delete proposal edits inside an isolated workspace."""
    root = Path(workspace_root).resolve()
    parsed = [
        {
            **changed_file,
            "path": str(changed_file["path"]),
            "operation": str(changed_file["operation"]),
            "content": str(changed_file["content"]),
        }
        for changed_file in changed_files
    ]
    destinations: list[Path] = []
    seen: set[Path] = set()
    for changed_file in parsed:
        relative = Path(changed_file["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PatchNormalizationError("Changed file path is not a safe relative path.")
        destination = (root / relative).resolve()
        if not destination.is_relative_to(root):
            raise PatchNormalizationError("Changed file path escapes the workspace.")
        if destination in seen:
            raise PatchNormalizationError("Proposal contains duplicate changed-file paths.")
        seen.add(destination)
        operation = changed_file["operation"]
        if operation not in {"modify", "add", "delete"}:
            raise PatchNormalizationError(f"Unsupported file operation: {operation}")
        if operation == "modify" and not destination.is_file():
            raise PatchNormalizationError(f"Cannot modify missing file: {relative}")
        if operation == "add" and destination.exists():
            raise PatchNormalizationError(f"Cannot add existing file: {relative}")
        if operation == "delete" and not destination.is_file():
            raise PatchNormalizationError(f"Cannot delete missing file: {relative}")
        destinations.append(destination)

    results: list[dict[str, Any]] = []
    for changed_file, destination in zip(parsed, destinations):
        operation = changed_file["operation"]
        relative = changed_file["path"]
        content = changed_file["content"]
        if operation == "add":
            lines = content.splitlines()
            match = _HUNK.match(lines[0]) if lines else None
            if match is None:
                raise PatchNormalizationError("Added files require one headerless unified-diff hunk.")
            old_start = int(match.group(1))
            old_count = int(match.group(2) or 1)
            new_start = int(match.group(3))
            new_count = int(match.group(4) or 1)
            if old_start != 0 or old_count != 0 or new_start != 1:
                raise PatchNormalizationError("Added-file hunk must start at an empty source.")
            if len(lines) < 2 or new_count != len(lines) - 1:
                raise PatchNormalizationError("Added-file hunk count does not match its body.")
            if any(not line.startswith("+") for line in lines[1:]):
                raise PatchNormalizationError("Added-file hunks may contain only added lines.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                "".join(line[1:] + "\n" for line in lines[1:]),
                encoding="utf-8",
                newline="",
            )
            results.append({"path": relative, "operation": operation, "hunks_applied": 1})
            continue

        if not content.strip():
            raise PatchNormalizationError(f"{operation.title()} patch is empty: {relative}")
        normalized, normalization = normalize_unified_diff(content, relative)
        application = apply_unified_diff(destination, normalized)
        if operation == "delete":
            if destination.read_text(encoding="utf-8").strip():
                raise PatchNormalizationError(
                    f"Delete patch did not remove all content from: {relative}"
                )
            destination.unlink()
        results.append(
            {
                "path": relative,
                "operation": operation,
                "hunks_applied": application["hunks_applied"],
                "normalization": normalization,
            }
        )
    return results
