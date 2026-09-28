"""CPU-only integration checks; no target model or GPU required."""
import contextlib
import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from privacy_tools import indirect_evaluation as e
from privacy_tools import indirect_data_check as d
from privacy_tools import unified as ui


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = {"results_dir": str(self.root / "results"),
                         "historical_results_dir": str(self.root / "history"),
                         "common": {"gpu": "2", "dataset": "sst2", "dataset_path": str(self.root / "data")},
                         "python": {"default": sys.executable, "indirect_lora": sys.executable, "indirect_icl": sys.executable}}
        self.capture = contextlib.redirect_stdout(io.StringIO())
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)

    def plan(self, model="chatglm", kind="explicit", dataset="sst2", **kwargs):
        return e.build_plan(self.settings, model, kind, dataset, self.root / "output", **kwargs)

    def write(self, file, value):
        file.parent.mkdir(parents=True, exist_ok=True)
        e.write_json(file, value)

    def fixture_metrics(self, plan):
        root = Path(plan["backend"])
        root.mkdir(parents=True, exist_ok=True)
        if plan["finetune_type"] == "explicit":
            columns = ("target_fpr", "actual_fpr", "tpr", "auc", "member_count", "nonmember_count", "tp", "fp", "tn", "fn", "threshold")
            with (root / "mia_evaluation_results.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=columns)
                w.writeheader()
                for target in e.TARGETS:
                    w.writerow(dict(target_fpr=target, actual_fpr=0, tpr=.5, auc=.75,
                                    member_count=1200, nonmember_count=1200, tp=600, fp=0, tn=1200, fn=600, threshold=.8))
        else:
            for field in e.FIELDS:
                self.write(root / field / "evaluation/metrics.json", {
                    "AUC": .75, "member_count": 10, "nonmember_count": 500,
                    **{f"TPR@{p}%": .5 for p in (100, 50, 10, 1, .1, .01)}})

    def source(self, kind):
        plan = self.plan(kind=kind)
        source = self.root / "source"
        summary = {**e.identity("chatglm", kind, "sst2"), "status": "completed",
                   "backend_digest": e.backend_digest(kind), "completed_at": "2026-09-27T00:00:00+00:00",
                   "output": str(source), "resources": {}, "data_validation": {"checked": True}}
        self.write(source / "summary.json", summary)
        backend = source / "backend"
        backend.mkdir()
        if kind == "explicit":
            for stem in ("member", "nonmember"):
                (backend / (stem + "_scores.pkl")).write_bytes(b"test")
        else:
            for field in e.FIELDS:
                (backend / field).mkdir()
                for split in ("other", "sst2"):
                    (backend / field / f"attack_results_{split}_bert_0_{field}.csv").write_text("diff_score\n1\n")
        return source

    def test_command_matrix_and_device_mapping(self):
        for model in ("chatglm", "fm9g"):
            for kind in ("explicit", "implicit"):
                for dataset in ("sst2", "wikitext2"):
                    with self.subTest(model=model, kind=kind, dataset=dataset):
                        plan = self.plan(model, kind, dataset)
                        cmd = plan["stages"][0][1]
                        self.assertEqual(cmd[0], sys.executable)
                        self.assertIn("evaluate/indirect_", cmd[1])
                        self.assertNotIn("glm/GLM_", " ".join(cmd))
                        self.assertEqual(plan["model"], "chatglm3" if model == "chatglm" else "fm9g")
                        self.assertEqual(plan["env"]["CUDA_VISIBLE_DEVICES"], "2")
                        self.assertEqual(cmd[cmd.index("--gpu")+1], "2" if kind == "explicit" else "0")
                        if kind == "implicit":
                            self.assertNotIn("--adapter-dir", cmd)
                            self.assertEqual(cmd[cmd.index("--num-test")+1], "20" if dataset == "sst2" else "0")
        self.assertFalse((self.root / "output").exists())

    def test_resource_overrides_and_legacy_migration(self):
        self.settings["methods"] = {"chatglm_lora": {"adapter_path": "/adapters/old"}}
        self.assertEqual(self.plan()["resources"]["adapter"], "/adapters/old")
        self.settings["evaluation"] = {"indirect": {"models": {"chatglm3": {"adapters": {"sst2": "/adapters/new"}, "base_model": "/base/configured"}}}}
        with patch.dict(os.environ, {"CHATGLM_MODEL_DIR": "/base/env"}):
            plan = self.plan()
            self.assertEqual(plan["resources"]["base_model"], "/base/env")
            self.assertEqual(plan["resources"]["adapter"], "/adapters/new")
        self.settings["common"]["icl_proposal_model"] = "/old/roberta-base"
        plan = self.plan(kind="implicit")
        self.assertTrue(plan["resources"]["neighbor_model"].endswith("bert-base-uncased"))
        self.assertTrue(plan["warnings"])

    def test_preview_and_history_never_execute_or_create_directories(self):
        with patch.object(e.subprocess, "run", side_effect=AssertionError("must not execute")):
            for kind in e.ROOTS:
                for model in ("chatglm", "fm9g"):
                    self.assertEqual(e.run_indirect(self.settings, model, kind, mode="preview"), 0)
                    self.assertEqual(e.run_indirect(self.settings, model, kind, mode="history"), 0)
        self.assertFalse((self.root / "results").exists())

    def test_history_filters_model_backend_and_status(self):
        ident = e.identity("fm9g", "implicit", "sst2")
        root = e.result_root(self.settings, ident)
        value = {**ident, "status": "completed", "backend_digest": e.backend_digest("implicit"), "completed_at": "2026-09-27", "results": {}, "output": "selected"}
        self.write(root / "valid/summary.json", value)
        for name, changes in (("failed", {"status":"failed"}), ("legacy", {"engine":"old"}), ("wrong", {"model":"chatglm3"})):
            self.write(root / name / "summary.json", {**value, **changes, "completed_at":"9999"})
        with patch.object(e, "print_summary") as display:
            self.assertEqual(e.show_history(self.settings, ident), 0)
            self.assertEqual(display.call_args.args[0]["output"], "selected")

    def test_invalid_gpu_and_combinations(self):
        self.settings["common"]["gpu"] = "0,1"
        with self.assertRaises(ValueError): self.plan()
        with self.assertRaises(ValueError): e.identity("unknown", "explicit", "sst2")

    def test_all_four_menus_route_before_legacy_method_resolution(self):
        for model in ("chatglm", "fm9g"):
            for kind in ("explicit", "implicit"):
                selection = ui.TaskSelection("evaluation", "none", "indirect_attack", "none", "indirect_attack", kind,
                     evaluation_model=model, evaluation_model_title=model, backend_method_id="nonexistent")
                with patch.object(ui, "_select_task", return_value=selection), patch.object(ui, "choose_from_menu", return_value="preview"), patch.object(e, "run_indirect", return_value=0) as run:
                    self.assertEqual(ui.interactive(self.settings), 0)
                    run.assert_called_once_with(self.settings, model, kind, mode="preview")

    def test_cancel_and_direct_placeholder_never_run(self):
        for model in ("chatglm", "fm9g"):
            selection = ui.TaskSelection("evaluation", "none", "direct_attack", "none", "direct_attack", "explicit", evaluation_model=model)
            with patch.object(e, "run_indirect", side_effect=AssertionError("direct placeholder")):
                self.assertEqual(ui._run_model_evaluation(selection, self.settings),0)
        with patch.object(ui, "_select_task", return_value=None):
            self.assertEqual(ui.interactive(self.settings),0)

    def test_native_lora_external_inputs_and_default_compatibility(self):
        spec=importlib.util.spec_from_file_location("lora_entry_test", e.ROOTS["explicit"] / "run.py")
        native=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        old=native.attack_command("fm9g","sst2","train",self.root)
        self.assertIn("--dataset-dir",old)
        cmd=native.attack_command("chatglm3","sst2","train",self.root,adapter="/a",neighbor_model="/n",dataset_file="/train.tsv",tokenizer="/t")
        for key,value in (("--adapter-path","/a"),("--neighbor-model-path","/n"),("--dataset-file","/train.tsv"),("--target-tokenizer-path","/t")):
            self.assertEqual(cmd[cmd.index(key)+1],value)
        self.assertNotIn("--dataset-dir",cmd)

    def test_offline_requires_matching_new_provenance(self):
        for kind in e.ROOTS:
            source=self.source(kind)
            plan=self.plan(kind=kind,source_run=source)
            self.assertNotIn("CUDA_VISIBLE_DEVICES",plan["env"])
            self.assertEqual(len(plan["stages"]),1 if kind=="explicit" else 4)
            with self.assertRaises(ValueError): self.plan(model="fm9g",kind=kind,source_run=source)
            import shutil
            shutil.rmtree(source)

    def test_success_and_failure_manifests(self):
        class Process:
            def __init__(self, command, returncode=0):
                self.stdout = io.StringIO("live output\n")
                self.command = command
                self.returncode = returncode

            def wait(self):
                return self.returncode

        for kind in e.ROOTS:
            def execute(command, **kwargs):
                latest=next((self.root / "results").rglob("run_manifest.json"))
                manifest=e.read_json(latest)
                self.fixture_metrics(manifest)
                return Process(command)
            with patch.object(e,"preflight"),patch.object(e.subprocess,"Popen",side_effect=execute):
                self.assertEqual(e.run_indirect(self.settings,"chatglm",kind),0)
            summary=e.read_json(next((self.root / "results").rglob("summary.json")))
            self.assertEqual(summary["status"],"completed")
            self.assertEqual(summary["data_validation"], {"status": "skipped"})
            self.assertEqual(next((self.root / "results").rglob("*.log")).read_text(), "live output\n")
            import shutil
            shutil.rmtree(self.root / "results")
        with patch.object(e,"preflight"),patch.object(e.subprocess,"Popen",return_value=Process([],9)) as runner:
            self.assertEqual(e.run_indirect(self.settings,"fm9g","implicit"),1)
            self.assertEqual(runner.call_count,1)
        self.assertFalse(list((self.root / "results").rglob("summary.json")))
        manifest=e.read_json(next((self.root / "results").rglob("run_manifest.json")))
        self.assertEqual(manifest["status"],"failed")

    def test_incomplete_metrics_do_not_succeed(self):
        for kind in e.ROOTS:
            plan=self.plan(kind=kind)
            self.fixture_metrics(plan)
            self.assertTrue(e.collect_results(plan))
            if kind=="implicit":
                file=Path(plan["backend"])/"synthetic_text/evaluation/metrics.json"
                metrics=e.read_json(file); metrics['member_count']=9; self.write(file,metrics)
                with self.assertRaises(ValueError): e.collect_results(plan)

    def test_missing_resources_and_bert_validation(self):
        with self.assertRaisesRegex(ValueError,"模型配置"): e.preflight(self.plan())
        plan=self.plan(kind="implicit")
        folder=Path(plan["resources"]["neighbor_model"])
        with patch.object(e,"require_model"),patch.object(e,"read_json",return_value={"model_type":"roberta"}):
            with self.assertRaisesRegex(ValueError,"BERT"): e.preflight(plan)

    def explicit_data(self):
        r=self.root
        adapter=r/'adapter';adapter.mkdir()
        (adapter/'adapter_model.safetensors').write_bytes(b'test adapter')
        for stem,prefix in (('member','training'),('nonmember','heldout')):
            with (r/(stem+'.tsv')).open('w',newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['sentence'])
                w.writerows([[f'{prefix} sentence {i}'] for i in range(1200)])
        self.write(adapter/'training_manifest.json',{'model':'chatglm3','dataset':'sst2','source':'test fixture',
            'adapter_sha256':d.digest(adapter/'adapter_model.safetensors'),
            'training_text_sha256':[d.text_key(f'training sentence {i}') for i in range(1200)]})
        return {'model':'chatglm3','dataset':'sst2','finetune_type':'explicit','resources':{
            'adapter':str(adapter),'training_manifest':str(adapter/'training_manifest.json'),
            'member_file':str(r/'member.tsv'),'nonmember_file':str(r/'nonmember.tsv')}}

    def test_membership_proof_and_disjointness(self):
        plan=self.explicit_data()
        self.assertEqual(d.validate_data(plan)['member_count'],1200)
        file=Path(plan['resources']['nonmember_file'])
        file.write_text(file.read_text().replace('heldout sentence 0','training sentence 0'))
        with self.assertRaisesRegex(ValueError,'交集'): d.validate_data(plan)

    def test_wrong_adapter_or_untrained_member_is_rejected(self):
        plan=self.explicit_data()
        file=Path(plan['resources']['member_file'])
        file.write_text(file.read_text().replace('training sentence 0','not trained sample'))
        with self.assertRaisesRegex(ValueError,'并非全部'): d.validate_data(plan)
        (Path(plan['resources']['adapter'])/'adapter_model.safetensors').write_bytes(b'wrong adapter')
        with self.assertRaisesRegex(ValueError,'指纹不匹配'): d.validate_data(plan)

    def test_real_subprocess_menu_preview(self):
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
        for model in ('1','2'):
            for kind,group in (('1','3'),('2','2')):
                result=subprocess.run([sys.executable,'-B',str(e.PROJECT_ROOT/'privacy_tool.py')],
                     input=f'2\n2\n{kind}\n{group}\n{model}\n3\n',text=True,capture_output=True,env=env)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('命令预览',result.stdout)
                self.assertIn('evaluate/indirect_',result.stdout)
                self.assertNotIn('glm/GLM_',result.stdout)


if __name__=='__main__':
    unittest.main()
