"""Create isolated Roslyn analysis projects for repositories without usable project documents."""

from __future__ import annotations

from pathlib import Path
import re
from xml.sax.saxutils import escape


def _safe_name(repository: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", repository)


def ensure_source_project(
    *,
    repository: str,
    root: Path,
    output: Path,
) -> Path:
    """Create a project outside the source tree that includes all C# files."""
    project = output / f"{_safe_name(repository)}.csproj"
    project.parent.mkdir(parents=True, exist_ok=True)
    if not project.is_file():
        source_glob = escape(str(root.resolve()).rstrip("\\") + r"\**\*.cs")
        project.write_text(
            "\n".join(
                [
                    '<Project Sdk="Microsoft.NET.Sdk">',
                    "  <PropertyGroup>",
                    "    <TargetFramework>net5.0</TargetFramework>",
                    "    <EnableDefaultCompileItems>false</EnableDefaultCompileItems>",
                    "    <EnableDefaultEmbeddedResourceItems>false</EnableDefaultEmbeddedResourceItems>",
                    "  </PropertyGroup>",
                    "  <ItemGroup>",
                    f'    <Compile Include="{source_glob}" />',
                    "  </ItemGroup>",
                    "</Project>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return project
