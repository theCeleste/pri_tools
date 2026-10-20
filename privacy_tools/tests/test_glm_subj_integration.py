"""CPU-only routing, command mapping, and history tests for Glm-subj."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from privacy_tools import glm_subj
from privacy_tools import unified as ui
from privacy_tools.core import Experiment, Method, run_experiment


class GlmSubjIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = {
            "common": {"gpu": "0", "chatglm3_6b": str(self.root / "chatglm3-6b")},
            "python": {"default": "python3", "glm_subj": "python3"},
            "glm_subj": {
                "base_model": str(self.root / "chatglm3-6b"),
                "raw_data_dir": str(self.root / "raw"),
                "embedding_data_dir": str(self.root / "embedding"),
                "output_dir": str(self.root / "outputs"),
                "prefer_latest_training_runs": True,
                "privsv_raw_adapter": str(self.root / "no-existing-raw-adapter"),
            },
        }

    def command_args(self, step):
        return step.command[2:]

    def test_finetune_pairings_use_only_glm_subj_entrypoints(self):
        run_dir = self.root / "run"
        expected = {
            "cliptext": ("/embedding", "/raw"),
            "zo_dpga": ("dpzero", "mezo"),
        }
        for method, (protected_data_or_method, baseline_data_or_method) in expected.items():
            with self.subTest(method=method):
                protected = glm_subj.finetune_steps(method, "protected", self.settings, run_dir)
                baseline = glm_subj.finetune_steps(method, "baseline", self.settings, run_dir)
                self.assertTrue(all(Path(step.command[1]).resolve() == glm_subj.ENTRY.resolve() for step in protected + baseline))
                if method == "cliptext":
                    self.assertIn(protected_data_or_method, protected[0].command[protected[0].command.index("--data-dir") + 1])
                    self.assertIn(baseline_data_or_method, baseline[0].command[baseline[0].command.index("--data-dir") + 1])
                    self.assertIn("embedding", " ".join(protected[0].command))
                else:
                    self.assertIn(protected_data_or_method, " ".join(protected[0].command))
                    self.assertIn(baseline_data_or_method, " ".join(baseline[0].command))
        private = glm_subj.finetune_steps("privsv", "protected", self.settings, run_dir)
        mean = glm_subj.finetune_steps("privsv", "baseline", self.settings, run_dir)
        self.assertEqual([step.command[2] for step in private], ["train-lora", "build-steering"])
        self.assertIn("private", " ".join(private[-1].command))
        self.assertIn("mean", " ".join(mean[-1].command))

    def test_direct_attack_pairs_are_glm_prefix_attack_and_compare(self):
        run_dir = self.root / "run"
        for method in ("cliptext", "privsv", "zo_dpga"):
            with self.subTest(method=method):
                steps = glm_subj.direct_attack_steps(method, self.settings, run_dir)
                self.assertEqual([step.command[2] for step in steps], ["prefix-attack", "prefix-attack", "compare"])
                self.assertTrue(all(Path(step.command[1]).resolve() == glm_subj.ENTRY.resolve() for step in steps))
                self.assertEqual(steps[0].command[steps[0].command.index("--data-dir") + 1],
                                 steps[1].command[steps[1].command.index("--data-dir") + 1])
                self.assertIn("--require-matched-targets", steps[-1].command)
                self.assertNotIn("causal-mia", " ".join(" ".join(step.command) for step in steps))

    def test_finetune_model_and_dataset_selection_enforces_supported_pair(self):
        settings = {"common": {"model_7b": "llama", "dataset": "sst2"},
                    "glm_subj": {"base_model": "chatglm3"}}
        with patch.object(ui, "choose_from_menu", side_effect=("chatglm3_6b", "subj")):
            self.assertEqual(ui._choose_model_and_dataset(settings, method_id="cliptext", experiment_id="utility"),
                             ("chatglm3-6b", "subj"))
        self.assertTrue(settings["_glm_subj_selected"])
        with patch.object(ui, "choose_from_menu", side_effect=("llama", "subj")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ui._choose_model_and_dataset(settings, method_id="privsv", experiment_id="utility"),
                             (None, None))

    def test_evaluation_selection_adds_chatglm_with_implicit_subj(self):
        answers = iter(("evaluation", "direct_attack", "explicit", "embedding", "cliptext", "chatglm_subj"))
        with patch.object(ui, "choose_from_menu", side_effect=lambda *_: next(answers)):
            selection = ui._select_task(self.settings)
        self.assertEqual(selection.method_id, "cliptext")
        self.assertEqual(selection.evaluation_model, "chatglm_subj")
        self.assertEqual(selection.dataset, "subj")

    def write_manifest(self, directory, **metadata):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run_manifest.json").write_text(json.dumps({
            "backend": "glm_subj", "method": "cliptext", "experiment": "utility",
            "model": "chatglm3-6b", "dataset": "subj", "role": "protected",
            "status": "completed", "created_at": "2026-10-01", "artifacts": {}, **metadata,
        }), encoding="utf-8")

    def test_history_reads_only_latest_matching_successful_training(self):
        root = Path(self.settings["glm_subj"]["output_dir"])
        success = root / "finetune" / "successful"
        report = success / "models" / "adapter" / "training_report.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"history": [{"validation": {"loss": 0.25, "perplexity": 1.28}}]}), encoding="utf-8")
        self.write_manifest(success, artifacts={"reports": [str(report)]})
        self.write_manifest(root / "wrong-method", method="zo_dpga")
        self.write_manifest(root / "failed", status="failed", created_at="2026-10-03")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            glm_subj.print_history(self.settings, "cliptext", "utility")
        self.assertIn("training_report.json", output.getvalue())
        self.assertNotIn("wrong-method", output.getvalue())
        self.assertNotIn("failed", output.getvalue())

    def test_history_falls_back_to_existing_glm_subj_outputs_without_manifests(self):
        root = Path(self.settings["glm_subj"]["output_dir"])
        report = root / "embedding_full_parallel/models/embedding/training_report.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"history": [{"epoch": 1, "validation": {"loss": 0.4}}]}), encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            glm_subj.print_history(self.settings, "cliptext", "utility")
        self.assertIn("旧结果无统一 run manifest", output.getvalue())
        self.assertIn("validation", output.getvalue())

    def test_direct_history_reads_manifest_summaries_and_comparison(self):
        root = Path(self.settings["glm_subj"]["output_dir"])
        run = root / "evaluation/cliptext/direct_attack/run-1"
        protected = run / "attacks/protected/prefix/summary.json"
        baseline = run / "attacks/baseline/prefix/summary.json"
        comparison = run / "attacks/comparison.json"
        for path in (protected, baseline):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "train": {"targets": 10, "fuzzy_success_rate": 0.2, "exact_success_rate": 0.1},
                "holdout": {"targets": 10, "fuzzy_success_rate": 0.0, "exact_success_rate": 0.0},
                "train_minus_holdout": {"fuzzy_success_rate": 0.2},
            }), encoding="utf-8")
        comparison.parent.mkdir(parents=True, exist_ok=True)
        comparison.write_text(json.dumps({"rows": [{"method": "ClipText+"}],
                                          "pairwise_comparison": {"operation": "second_minus_first"}}),
                              encoding="utf-8")
        self.write_manifest(run, experiment="direct_attack", artifacts={
            "summaries": [str(protected), str(baseline)], "comparison": str(comparison),
        })
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            glm_subj.print_history(self.settings, "cliptext", "direct_attack")
        self.assertEqual(output.getvalue().count("train: targets=10"), 2)
        self.assertIn("pairwise_comparison", output.getvalue())

    def test_direct_attack_uses_latest_successful_finetune_artifacts(self):
        root = Path(self.settings["glm_subj"]["output_dir"])
        run = root / "finetune/cliptext/utility/run-1"
        protected_adapter = run / "models/cliptext"
        protected_adapter.mkdir(parents=True)
        (protected_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        self.write_manifest(run, artifacts={"model_outputs": [str(protected_adapter)]})
        protected, _, _, _ = glm_subj.direct_model_paths(self.settings, "cliptext")
        self.assertEqual(protected, protected_adapter)

    def test_manifest_status_distinguishes_preview_from_success(self):
        output_root = self.root / "manifests"
        method = Method("cliptext", "ClipText", "test", glm_subj.GLM_ROOT, [],
                        [Experiment("utility", "test", "test", lambda *_: [])])
        settings = {"results_dir": str(output_root), "_manifest_metadata": glm_subj.manifest_metadata("cliptext", "utility")}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_experiment("finetune", method, method.experiments[0], settings, dry_run=True), 0)
            self.assertEqual(run_experiment("finetune", method, method.experiments[0], settings, dry_run=False), 0)
        manifests = list(output_root.rglob("run_manifest.json"))
        self.assertEqual({json.loads(item.read_text())["status"] for item in manifests}, {"previewed", "completed"})

    def test_unimplemented_paths_do_not_call_legacy_backends_or_succeed(self):
        evaluation = ui.TaskSelection("evaluation", "cliptext", "indirect_attack", "embedding",
                                       "indirect_attack", "explicit", dataset="subj",
                                       evaluation_model="chatglm_subj")
        finetune = ui.TaskSelection("finetune", "zo_dpga", "", "gradient")
        with patch.object(ui, "choose_from_menu", return_value="run"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ui._run_glm_subj_task(evaluation, self.settings), 2)
            self.assertEqual(ui._run_glm_subj_task(finetune, self.settings, experiment_id="performance_baseline"), 2)


if __name__ == "__main__":
    unittest.main()
