"""History archive checks for the unified direct evaluator."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from privacy_tools import direct_evaluation as e
from privacy_tools import unified as ui


class DirectHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = {
            "results_dir": str(root / "results"),
            "historical_results_dir": str(root / "history"),
        }
        self.ident = e.identity("chatglm", "explicit", "sst2")

    def summary(self, output="run-1"):
        matching = {"candidate_count": 100, "matched_count": 4, "match_rate": 0.04}
        return {
            **self.ident,
            "schema_version": 1,
            "baseline_type": "no_perturbation",
            "evaluation_type": "direct",
            "status": "completed",
            "updated_at": "2026-09-29T00:00:00+00:00",
            "backend_digest": "digest",
            "matching": matching,
            "results": matching,
            "output": output,
        }

    def test_latest_result_overwrites_previous_result(self):
        e.archive_result(self.settings, "direct", self.summary("old"), "旧结果")
        e.archive_result(self.settings, "direct", self.summary("new"), "新结果")
        history = e.load_result(self.settings, "direct", self.ident)
        self.assertEqual(history["kind"], "summary")
        self.assertEqual(history["summary"]["output"], "new")
        target = Path(self.settings["historical_results_dir"]) / "no_perturbation/direct/chatglm3/explicit/sst2"
        self.assertEqual((target / "result.txt").read_text(encoding="utf-8"), "新结果")
        self.assertEqual(len(list(target.iterdir())), 2)

    def test_summary_uses_evaluation_method_and_chinese_finetune_type(self):
        text = e.format_summary(self.summary())
        self.assertTrue(text.startswith(
            "方法：直接评估 模型：chatglm3，微调方式：显式，数据集：sst2"
        ))
        implicit = self.summary()
        implicit["finetune_type"] = "implicit"
        self.assertIn("微调方式：隐式", e.format_summary(implicit))

    def test_history_reads_text_verbatim_when_summary_is_invalid(self):
        target = Path(self.settings["historical_results_dir"]) / "no_perturbation/direct/chatglm3/explicit/sst2"
        target.mkdir(parents=True)
        (target / "summary.json").write_text("[]", encoding="utf-8")
        text = "原样第一行\n原样第二行"
        (target / "result.txt").write_text(text, encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(e.show_history(self.settings, self.ident), 0)
        self.assertEqual(output.getvalue(), text)

    def test_history_does_not_read_legacy_results_directory(self):
        legacy = Path(self.settings["results_dir"]) / "direct_evaluation/chatglm3/explicit/sst2/old"
        legacy.mkdir(parents=True)
        (legacy / "summary.json").write_text("{}", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(e.show_history(self.settings, self.ident), 0)
        self.assertIn("未找到", output.getvalue())

    def test_menu_route_passes_selected_dataset_to_direct_evaluation(self):
        selection = ui.TaskSelection(
            "evaluation", "none", "direct_attack", "none", "direct_attack", "explicit",
            dataset="wikitext2", evaluation_model="chatglm", evaluation_model_title="ChatGLM3-6b",
        )
        output = io.StringIO()
        with patch.object(ui, "choose_from_menu", return_value="preview"), \
                patch.object(e, "run_direct", return_value=0) as run, \
                contextlib.redirect_stdout(output):
            self.assertEqual(ui._run_model_evaluation(selection, self.settings), 0)
        run.assert_called_once_with(
            self.settings, "chatglm", "explicit", mode="preview", dataset="wikitext2",
        )
        self.assertIn(
            "方法：直接评估 模型：chatglm3，微调方式：显式，数据集：wikitext2",
            output.getvalue(),
        )

    def test_wikitext_json_snapshot_passes_preflight_and_loads_train_only(self):
        dataset = Path(self.temp.name) / "wikitext2"
        dataset.mkdir()
        (dataset / "dataset_config.json").write_text('{"dataset":"wikitext2"}', encoding="utf-8")
        (dataset / "train.json").write_text(
            json.dumps(["training paragraph", {"text": "second training paragraph"}]),
            encoding="utf-8",
        )
        (dataset / "validation.json").write_text(json.dumps(["held-out paragraph"]), encoding="utf-8")
        (dataset / "test.json").write_text(json.dumps(["test paragraph"]), encoding="utf-8")

        e.require_dataset("wikitext2", dataset)
        self.assertEqual(
            [row["text"] for row in e.load_training_records("wikitext2", dataset)],
            ["training paragraph", "second training paragraph"],
        )

        script = e.ROOT / "DPICL/member_test_current.py"
        spec = importlib.util.spec_from_file_location("direct_member_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            [row["text"] for row in module.load_text_records(str(dataset), "wikitext")],
            ["training paragraph", "second training paragraph"],
        )

    def test_wikitext_metadata_without_training_data_is_rejected(self):
        dataset = Path(self.temp.name) / "wikitext2-empty"
        dataset.mkdir()
        (dataset / "dataset_config.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "train.json"):
            e.require_dataset("wikitext2", dataset)
