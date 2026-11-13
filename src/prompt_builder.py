"""Build structured prompts for Feature Envy refactoring experiments."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def prompt_sha256(prompt: str | bytes) -> str:
    """Hash logical UTF-8 prompt content, reversing Windows text-mode writes."""
    if isinstance(prompt, bytes):
        prompt = prompt.replace(b"\r\n", b"\n").decode("utf-8")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_refactoring_prompt(
    *,
    target: dict[str, Any],
    target_source: str,
    target_location: str,
    host_class_info: str,
    retrieved_documents: Iterable[dict[str, Any]],
    dependency_context: dict[str, Any] | None = None,
) -> str:
    """Build a prompt with optional Roslyn dependency evidence."""
    context_sections = []
    for rank, document in enumerate(retrieved_documents, start=1):
        context_sections.append(
            "\n".join(
                [
                    f"### Retrieved method {rank}",
                    f"Document ID: {document['DocumentId']}",
                    f"Repository: {document['Repository']}",
                    f"File: {document['File']}",
                    f"Location: lines {document['StartLine']}-{document['EndLine']}",
                    f"Containing type: {document['ContainingType']}",
                    f"Method: {document['MethodName']}",
                    f"Signature: {document['Signature']}",
                    "Source code:",
                    "```csharp",
                    document.get("SourceCode", ""),
                    "```",
                ]
            )
        )

    sections = [
        "## 1. Refactoring task",
        (
            "Refactor the C# target method to reduce inappropriate dependence on "
            "members of another class while preserving existing functionality, "
            "observable behavior, and public API compatibility. Choose a suitable "
            "refactoring, such as Move Method or Extract Method, when justified; "
            "do not force a particular transformation."
        ),
        (
            "Use only real repository symbols where possible. Identify every file "
            "requiring changes, including callers or related code when applicable. "
            "Do not assume or invent unavailable classes, methods, callers, or APIs. "
            "If the available context is insufficient to propose a safe change, "
            "state that clearly instead of guessing."
        ),
        "## 2. Original target method",
        f"Repository: {target['Repository']}",
        f"Commit: {target['Commit']}",
        f"Method identifier: {target['MethodIdentifier']}",
        f"Source location: {target_location}",
        "Original target code:",
        "```csharp",
        target_source,
        "```",
        "## 3. Fixed host-class information",
        host_class_info,
        "## 4. Retrieved repository context",
        (
            "The following ranked methods are repository context, not the target "
            "method or an expected solution. Use their source and metadata only "
            "when relevant to the proposed change."
        ),
        "\n\n".join(context_sections),
    ]
    if dependency_context is not None:
        sections.extend(
            [
                "## 5. Roslyn-derived dependency information",
                (
                    "This evidence describes semantic dependencies of the target "
                    "method. The retrieved repository methods above are unchanged "
                    "from the paired baseline condition. Use only the evidence shown; "
                    "do not infer unresolved dependencies."
                ),
                "```json",
                json.dumps(dependency_context, indent=2, sort_keys=True),
                "```",
            ]
        )
    sections.extend(
        [
            f"## {6 if dependency_context is not None else 5}. Required output format",
            (
                "Return exactly one JSON object and no Markdown fences or additional "
                "text. Use this schema: "
                '{"status":"READY"|"INSUFFICIENT_CONTEXT",'
                '"proposal":{"summary":"string","rationale":"string",'
                '"changed_files":[{"path":"repository-relative string",'
                '"operation":"modify"|"add"|"delete",'
                '"edit":{"type":"unified_diff","content":"string"}}],'
                '"validation_plan":["string"]},'
                '"insufficient_context":{"reason":"string","missing":["string"]}}. '
                "For READY, provide a concrete proposal and set "
                '"insufficient_context" to null. For INSUFFICIENT_CONTEXT, set '
                '"proposal" to null and explain the missing evidence. The edit content '
                "must be a precise, repository-relative unified diff or an empty string "
                "only when no edit is proposed. Do not claim that tests passed unless "
                "they were actually run."
            ),
        ]
    )
    return "\n\n".join(sections)
