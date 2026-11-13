import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.patch_normalizer import (
    PatchNormalizationError,
    apply_changed_files,
    parse_changed_files,
)
from src.hybrid_retriever import RRF_K, _fuse


class PatchSchemaTests(unittest.TestCase):
    def test_reads_nested_changed_file_edit(self):
        response = {
            "proposal": {
                "changed_files": [
                    {
                        "path": "src/Example.cs",
                        "operation": "modify",
                        "edit": {"type": "unified_diff", "content": "@@ -1 +1 @@"},
                    }
                ]
            }
        }
        self.assertEqual(
            parse_changed_files(response),
            [
                {
                    "path": "src/Example.cs",
                    "operation": "modify",
                    "content": "@@ -1 +1 @@",
                }
            ],
        )

    def test_rejects_flat_patch_schema(self):
        response = {
            "proposal": {
                "changed_files": [
                    {"path": "src/Example.cs", "operation": "modify", "unified_diff": "..."}
                ]
            }
        }
        with self.assertRaises(PatchNormalizationError):
            parse_changed_files(response)


class HybridRetrievalTests(unittest.TestCase):
    def test_rrf_fuses_rankings_deterministically(self):
        first = [{"DocumentId": "a"}, {"DocumentId": "b"}]
        second = [{"DocumentId": "b"}, {"DocumentId": "c"}]
        result = _fuse(first, second, top_k=3)
        self.assertEqual([hit["DocumentId"] for hit in result], ["b", "a", "c"])
        self.assertAlmostEqual(
            result[0]["RRFScore"],
            1 / (RRF_K + 1) + 1 / (RRF_K + 2),
        )

    def test_rejects_unsafe_path(self):
        response = {
            "proposal": {
                "changed_files": [
                    {
                        "path": "../Example.cs",
                        "operation": "modify",
                        "edit": {"type": "unified_diff", "content": ""},
                    }
                ]
            }
        }
        with self.assertRaises(PatchNormalizationError):
            parse_changed_files(response)

    def test_applies_modify_add_and_delete_operations(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "Existing.cs"
            existing.write_text("before\n", encoding="utf-8")
            (root / "Delete.cs").write_text("delete me\n", encoding="utf-8")
            results = apply_changed_files(
                root,
                [
                    {
                        "path": "Existing.cs",
                        "operation": "modify",
                        "content": "@@ -1 +1 @@\n-before\n+after",
                    },
                    {
                        "path": "Added.cs",
                        "operation": "add",
                        "content": "@@ -0,0 +1,1 @@\n+class Added {}",
                    },
                    {
                        "path": "Delete.cs",
                        "operation": "delete",
                        "content": "@@ -1 +0,0 @@\n-delete me",
                    },
                ],
            )
            self.assertEqual(existing.read_text(encoding="utf-8"), "after\n")
            self.assertEqual(
                (root / "Added.cs").read_text(encoding="utf-8"),
                "class Added {}\n",
            )
            self.assertFalse((root / "Delete.cs").exists())
            self.assertEqual([item["operation"] for item in results], ["modify", "add", "delete"])

    def test_rejects_adding_an_existing_file(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Existing.cs").write_text("unchanged\n", encoding="utf-8")
            with self.assertRaises(PatchNormalizationError):
                apply_changed_files(
                    root,
                    [
                        {
                            "path": "Existing.cs",
                            "operation": "add",
                            "content": "@@ -0,0 +1,1 @@\n+class Added {}",
                        }
                    ],
                )


if __name__ == "__main__":
    unittest.main()
