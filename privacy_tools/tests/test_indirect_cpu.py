"""Real native offline subprocess checks using synthetic scores, never model inference."""
import contextlib
import csv
import importlib.util
import io
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

from privacy_tools import indirect_evaluation as e

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in
                ("torch", "numpy", "pandas", "sklearn", "matplotlib"))


@unittest.skipUnless(AVAILABLE, "Run with a backend Python containing CPU evaluation dependencies")
class NativeOfflineTests(unittest.TestCase):
    def test_four_model_routes_recompute_real_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = {"results_dir": str(root / "results"),
                        "historical_results_dir": str(root / "history"),
                        "common": {"gpu": "not-needed"},
                        "python": {"default": sys.executable, "indirect_lora": sys.executable,
                                   "indirect_icl": sys.executable}}
            for model in ("chatglm3", "fm9g"):
                for kind in e.ROOTS:
                    with self.subTest(model=model, kind=kind):
                        source = root / (model + "_" + kind)
                        backend = source / "backend"
                        backend.mkdir(parents=True)
                        ident = e.identity(model, kind, "sst2")
                        e.write_json(source / "summary.json", {
                            **ident, "backend_digest": e.backend_digest(kind), "status": "completed",
                            "resources": {}, "data_validation": {"test_fixture": True}})
                        if kind == "explicit":
                            for split, values in (("member", (.9, .7)), ("nonmember", (.8, .1))):
                                records = [{"<original_text>: x": values[i % 2], ("neighbour", .5): 0.0}
                                           for i in range(1200)]
                                with (backend / (split + "_scores.pkl")).open("wb") as stream:
                                    pickle.dump(records, stream)
                        else:
                            for field in e.FIELDS:
                                (backend / field).mkdir()
                                for split, count, values in (("other", 10, (.9, .7)), ("sst2", 500, (.8, .1))):
                                    file = backend / field / f"attack_results_{split}_bert_0_{field}.csv"
                                    with file.open("w", newline="") as stream:
                                        writer = csv.writer(stream)
                                        writer.writerow(["diff_score"])
                                        writer.writerows([[values[i % 2]] for i in range(count)])
                        source_before = {str(f): f.read_bytes() for f in source.rglob("*") if f.is_file()}
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output):
                            code = e.run_indirect(settings, model, kind, dataset="sst2", mode="evaluate", source_run=source)
                        self.assertEqual(code, 0, output.getvalue())
                        summaries = list(e.result_root(settings, ident).glob("*/summary.json"))
                        self.assertEqual(len(summaries), 1)
                        summary = e.read_json(summaries[0])
                        self.assertEqual(summary["data_validation"], {"test_fixture": True})
                        self.assertEqual(summary["source_run"], str(source))
                        results = summary["results"]
                        self.assertEqual(set(results), {"lora"} if kind == "explicit" else set(e.FIELDS))
                        for metrics in results.values():
                            self.assertAlmostEqual(metrics.get("auc", metrics.get("AUC")), .75)
                            self.assertEqual(metrics["member_count"], 1200 if kind == "explicit" else 10)
                            self.assertEqual(metrics["nonmember_count"], 1200 if kind == "explicit" else 500)
                        manifest = e.read_json(summaries[0].parent / "run_manifest.json")
                        self.assertEqual(manifest["status"], "completed")
                        self.assertEqual(len(manifest["stage_results"]), 1 if kind == "explicit" else 2)
                        self.assertTrue(all(stage["returncode"] == 0 for stage in manifest["stage_results"]))
                        self.assertEqual(source_before, {str(f): f.read_bytes() for f in source.rglob("*") if f.is_file()})
                        with contextlib.redirect_stdout(io.StringIO()) as history:
                            self.assertEqual(e.run_indirect(settings, model, kind, mode="history", dataset="sst2"), 0)
                        self.assertIn("AUC=0.750000", history.getvalue())


if __name__ == "__main__":
    unittest.main()
