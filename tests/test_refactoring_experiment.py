import unittest
from unittest.mock import patch
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_refactoring_experiment import (
    _new_validation_attempt,
    _new_validation_workspace,
    _saved_run_for_prompt,
    _validate_job,
    _write_json,
)
from deepseek_client import DeepSeekConfig
from deepseek_experiment import run_experiment
from prompt_builder import prompt_sha256


class ValidationAttemptTests(unittest.TestCase):
    def test_incomplete_validation_attempt_is_preserved_and_retried_separately(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = _new_validation_attempt(output, "bm25/baseline/example")
            (first / "partial.log").write_text("preserve", encoding="utf-8")

            second = _new_validation_attempt(output, "bm25/baseline/example")

            self.assertEqual(first.name, "attempt-001")
            self.assertEqual(second.name, "attempt-002")
            self.assertEqual(
                (first / "partial.log").read_text(encoding="utf-8"),
                "preserve",
            )

    def test_short_workspaces_are_unique_and_preserve_existing_attempts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            workspace_root = root / "work"

            first = _new_validation_workspace(source, str(workspace_root))
            first.mkdir()
            (first / "partial.log").write_text("preserve", encoding="utf-8")
            second = _new_validation_workspace(source, str(workspace_root))

            self.assertNotEqual(first, second)
            self.assertEqual(first.name, "w")
            self.assertEqual(second.name, "w")
            self.assertEqual(first.parent.parent, workspace_root.resolve())
            self.assertFalse(second.exists())
            self.assertEqual(
                (first / "partial.log").read_text(encoding="utf-8"), "preserve"
            )

    def test_workspace_root_inside_pinned_source_is_rejected(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "must not be inside"):
                _new_validation_workspace(source, str(source / "work"))
            self.assertFalse((source / "work").exists())

    def test_validation_applies_patch_only_in_short_workspace(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "Project.csproj").write_text("<Project />", encoding="utf-8")
            (source / "Target.cs").write_text("before\n", encoding="utf-8")
            metadata = root / "revision.json"
            metadata.write_text(json.dumps({"sha": "commit"}), encoding="utf-8")
            baseline = root / "baseline.json"
            baseline.write_text("{}", encoding="utf-8")
            job = {
                "Repository": "owner/repo",
                "Commit": "commit",
                "SourcePath": "Target.cs",
                "CaseId": "Example.Target.Method()",
                "BaselineAnalysisPath": str(baseline),
            }
            proposal = {
                "proposal": {
                    "changed_files": [{
                        "path": "Target.cs",
                        "operation": "modify",
                        "edit": {
                            "type": "unified_diff",
                            "content": "@@ -1 +1 @@\n-before\n+after",
                        },
                    }]
                }
            }
            config = {
                "revision_metadata": str(metadata),
                "source_root": str(source),
                "source_commit": "commit",
                "workspace_root": str(root / "short"),
                "project": "Project.csproj",
                "analyzer": "analyzer.exe",
                "msbuild": "sdk",
                "restore_commands": [["restore"]],
            }
            output = root / "long-output" / "attempt"
            output.mkdir(parents=True)
            analysis = {
                "Repository": {"Commit": "commit"},
                "Target": {"Signature": job["CaseId"]},
            }
            with patch("run_refactoring_experiment.invoke_roslyn", return_value=analysis), \
                 patch(
                     "run_refactoring_experiment._run_commands",
                     return_value={"status": "PASS"},
                 ) as commands:
                result = _validate_job(
                    job=job, proposal=proposal, config=config, output_directory=output
                )

            workspace = Path(result["workspace"])
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["restore"]["status"], "PASS")
            self.assertEqual(commands.call_count, 3)
            self.assertEqual(commands.call_args_list[0].kwargs["commands"], [["restore"]])
            self.assertEqual(workspace.parent.parent, (root / "short").resolve())
            self.assertEqual((workspace / "Target.cs").read_text(encoding="utf-8"), "after\n")
            self.assertEqual((source / "Target.cs").read_text(encoding="utf-8"), "before\n")

    def test_failed_restore_blocks_roslyn_and_build(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "Project.csproj").write_text("<Project />", encoding="utf-8")
            (source / "Target.cs").write_text("before\n", encoding="utf-8")
            metadata = root / "revision.json"
            metadata.write_text(json.dumps({"sha": "commit"}), encoding="utf-8")
            job = {
                "Repository": "owner/repo",
                "Commit": "commit",
                "SourcePath": "Target.cs",
                "CaseId": "Example.Target.Method()",
            }
            proposal = {
                "proposal": {
                    "changed_files": [{
                        "path": "Target.cs",
                        "operation": "modify",
                        "edit": {
                            "type": "unified_diff",
                            "content": "@@ -1 +1 @@\n-before\n+after",
                        },
                    }]
                }
            }
            config = {
                "revision_metadata": str(metadata),
                "source_root": str(source),
                "source_commit": "commit",
                "workspace_root": str(root / "short"),
                "project": "Project.csproj",
                "restore_commands": [["restore"]],
            }
            output = root / "attempt"
            output.mkdir()
            with patch("run_refactoring_experiment.invoke_roslyn") as analyzer, \
                 patch(
                     "run_refactoring_experiment._run_commands",
                     return_value={"status": "FAIL"},
                 ) as commands:
                result = _validate_job(
                    job=job, proposal=proposal, config=config, output_directory=output
                )

            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(result["restore"]["status"], "FAIL")
            commands.assert_called_once()
            analyzer.assert_not_called()
            self.assertEqual((source / "Target.cs").read_text(encoding="utf-8"), "before\n")


class AtomicCheckpointWriteTests(unittest.TestCase):
    def test_write_retries_transient_windows_replace_error_and_preserves_old_temp(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "checkpoint.json"
            checkpoint.write_text('{"old": true}', encoding="utf-8")
            stale_temp = directory / "checkpoint.json.tmp"
            stale_temp.write_text("preserve", encoding="utf-8")
            real_replace = __import__("os").replace
            attempts = 0

            def replace_with_transient_lock(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError("transient sharing violation")
                real_replace(source, destination)

            with patch("run_refactoring_experiment.os.replace", replace_with_transient_lock):
                _write_json(checkpoint, {"new": True})

            self.assertEqual(attempts, 2)
            self.assertEqual(checkpoint.read_text(encoding="utf-8"), '{\n  "new": true\n}')
            self.assertEqual(stale_temp.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(
                [path.name for path in directory.glob(".*.tmp")],
                [],
            )

    def test_write_failure_preserves_existing_checkpoint_and_cleans_own_temp(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "checkpoint.json"
            checkpoint.write_text('{"old": true}', encoding="utf-8")
            stale_temp = directory / "checkpoint.json.tmp"
            stale_temp.write_text("preserve", encoding="utf-8")

            with patch(
                "run_refactoring_experiment.os.replace",
                side_effect=PermissionError("replacement remains denied"),
            ), patch("run_refactoring_experiment.time.sleep"):
                with self.assertRaises(PermissionError):
                    _write_json(checkpoint, {"new": True})

            self.assertEqual(checkpoint.read_text(encoding="utf-8"), '{"old": true}')


class ResumableGenerationTests(unittest.TestCase):
    def test_saved_response_callback_runs_after_atomic_response_write(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt", encoding="utf-8")
            observed = []

            def response_saved(directory):
                response_path = directory / "response.json"
                self.assertTrue(response_path.is_file())
                observed.append(json.loads(response_path.read_text(encoding="utf-8")))

            run_directory = run_experiment(
                prompt_path=prompt,
                output_root=root / "runs",
                config=DeepSeekConfig(),
                live=False,
                response_saved=response_saved,
            )

            self.assertEqual(len(observed), 1)
            self.assertTrue((run_directory / "response.json").is_file())

    def test_orphaned_response_is_recovered_by_prompt_and_model_identity(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_directory = root / "bm25-deepseek-orphaned"
            run_directory.mkdir()
            prompt = run_directory / "prompt.txt"
            prompt.write_text("prompt", encoding="utf-8")
            config = {
                "mode": "LIVE",
                "model": "deepseek-flash",
                "endpoint": DeepSeekConfig.endpoint,
                "temperature": 0.0,
                "max_tokens": 4096,
                "live_request": True,
            }
            (run_directory / "model-config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            (run_directory / "response.json").write_text(
                json.dumps({"choices": []}),
                encoding="utf-8",
            )

            recovered = _saved_run_for_prompt(
                prompt_sha256("prompt"),
                root,
                {
                    "model": "deepseek-flash",
                    "endpoint": DeepSeekConfig.endpoint,
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
            )

            self.assertEqual(recovered, run_directory)


if __name__ == "__main__":
    unittest.main()
