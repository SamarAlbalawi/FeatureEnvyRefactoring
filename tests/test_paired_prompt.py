import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_generation_jobs import (
    _canonical_signature,
    _retrieval_identity_error,
    _target_document,
    _write_prepared_prompt,
)
from prompt_builder import build_refactoring_prompt, prompt_sha256


class PairedPromptTests(unittest.TestCase):
    def setUp(self):
        self.common = {
            "target": {
                "Repository": "owner/repo",
                "Commit": "abc123",
                "MethodIdentifier": "Example.Host.Target()",
            },
            "target_source": "void Target() {}",
            "target_location": "src/Host.cs#L1-L1",
            "host_class_info": "Containing type: Example.Host",
            "retrieved_documents": [
                {
                    "DocumentId": "doc-1",
                    "Repository": "owner/repo",
                    "File": "src/Other.cs",
                    "StartLine": 2,
                    "EndLine": 3,
                    "ContainingType": "Example.Other",
                    "MethodName": "Helper",
                    "Signature": "Example.Other.Helper()",
                    "SourceCode": "void Helper() {}",
                }
            ],
        }

    def test_dependency_context_is_the_only_pairwise_prompt_difference(self):
        baseline = build_refactoring_prompt(**self.common)
        augmented = build_refactoring_prompt(
            **self.common,
            dependency_context={"Fields": [{"Symbol": "Example.Other.Value"}]},
        )
        self.assertIn("Document ID: doc-1", baseline)
        self.assertIn("Document ID: doc-1", augmented)
        self.assertNotIn("Roslyn-derived dependency information", baseline)
        self.assertIn("Roslyn-derived dependency information", augmented)
        self.assertIn('"Symbol": "Example.Other.Value"', augmented)
        self.assertEqual(
            baseline.split("## 5. Required output format")[0],
            augmented.split("## 5. Roslyn-derived dependency information")[0],
        )

    def test_prepared_prompt_can_resume_without_overwriting(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline" / "prompt.txt"
            _write_prepared_prompt(path, "stable prompt")
            _write_prepared_prompt(path, "stable prompt")

            self.assertEqual(path.read_text(encoding="utf-8"), "stable prompt")
            with self.assertRaises(FileExistsError):
                _write_prepared_prompt(path, "different prompt")

    def test_prompt_hash_handles_lf_and_embedded_crlf_file_writes(self):
        for prompt in (
            "first line\nsecond line\n",
            "retrieved source\r\nwith embedded CRLF\nfollowed by LF\n",
        ):
            with self.subTest(prompt=prompt):
                windows_text_file = prompt.encode("utf-8").replace(
                    b"\n",
                    b"\r\n",
                )
                self.assertEqual(
                    prompt_sha256(prompt),
                    prompt_sha256(windows_text_file),
                )

    def test_retrieval_result_requires_exact_case_revision_and_target(self):
        case = {
            "CaseId": "Example.Host.Target()",
            "Repository": "owner/repo",
            "Commit": "abc123",
        }
        target = {
            "DocumentId": "target-doc",
            "Signature": case["CaseId"],
            "Repository": case["Repository"],
            "Commit": case["Commit"],
        }
        result = {
            "QueryTargetSignature": case["CaseId"],
            "QueryTargetDocumentId": "target-doc",
        }
        self.assertIsNone(_retrieval_identity_error(result, case, target))
        self.assertIsNone(
            _retrieval_identity_error(
                {"QueryTargetDocumentId": "target-doc"},
                case,
                target,
            )
        )
        self.assertIn(
            "CaseId",
            _retrieval_identity_error(
                {**result, "QueryTargetSignature": "Example.Host.Other()"},
                case,
                target,
            ),
        )
        self.assertIn(
            "pinned corpus target",
            _retrieval_identity_error(
                {**result, "QueryTargetDocumentId": "other-doc"},
                case,
                target,
            ),
        )
        self.assertIn(
            "exact case, repository, and commit",
            _retrieval_identity_error(
                result,
                case,
                {**target, "Commit": "different"},
            ),
        )

    def test_signature_aliases_match_semantically_without_loosening_identity(self):
        case_signature = (
            "Example.Host.Handle(Dictionary<Guid, Example.Item>, "
            "IReadOnlyList<object>, DateTime, Type)"
        )
        roslyn_signature = (
            "Example.Host.Handle(System.Collections.Generic.Dictionary<"
            "System.Guid, Example.Item>, "
            "System.Collections.Generic.IReadOnlyList<System.Object>, "
            "System.DateTime, System.Type)"
        )
        self.assertEqual(
            _canonical_signature(case_signature),
            _canonical_signature(roslyn_signature),
        )
        case = {
            "CaseId": case_signature,
            "Repository": "owner/repo",
            "Commit": "abc123",
            "File": "src/Host.cs",
        }
        target = {
            "DocumentId": "exact-target",
            "Signature": roslyn_signature,
            "Repository": "owner/repo",
            "Commit": "abc123",
            "File": "src/Host.cs",
        }
        self.assertIs(
            _target_document({"exact-target": target}, case),
            target,
        )
        with self.assertRaisesRegex(ValueError, "found 0"):
            _target_document(
                {
                    "wrong-revision": {
                        **target,
                        "DocumentId": "wrong-revision",
                        "Commit": "different",
                    }
                },
                case,
            )
        self.assertIsNone(
            _retrieval_identity_error(
                {
                    "QueryTargetSignature": roslyn_signature,
                    "QueryTargetDocumentId": "exact-target",
                },
                case,
                target,
            )
        )
        self.assertIsNotNone(
            _retrieval_identity_error(
                {
                    "QueryTargetSignature": "Example.Host.Other()",
                    "QueryTargetDocumentId": "exact-target",
                },
                case,
                target,
            )
        )


if __name__ == "__main__":
    unittest.main()
