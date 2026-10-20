"""CPU-only integration contracts for the isolated ChatGLM/Subj backend."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from privacy_tools import glm_subj as g, unified as ui
from privacy_tools.bootstrap import environment_plan


class GlmSubjTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = {'python': {'glm_subj': 'isolated-python'}, 'glm_subj': {
            'output_root': str(self.root), 'raw_data': '/raw-subj', 'embedding_data': '/clip-subj',
            'base_model': '/chatglm', 'artifacts': {}}}

    def build(self, method, experiment):
        return g.build_steps(self.settings, self.root / 'new', method, experiment)

    def manifest(self, method='cliptext', experiment='utility', status='success',
                 completed='2026-01-01T00:00:00+00:00', name='old', **extra):
        directory = self.root / name
        directory.mkdir()
        report = directory / 'report.json'
        report.write_text('{}')
        value = {**g.identity(method, experiment), 'status': status, 'completed_at': completed,
                 'artifacts': {'report': str(report)}, 'products': {'adapter': str(directory)}, **extra}
        (directory / 'run_manifest.json').write_text(json.dumps(value))
        return value

    def test_training_mapping_and_parser_compatibility(self):
        for method in g.METHODS:
            for exp in ('utility', 'baseline'):
                steps, artifacts, products = self.build(method, exp)
                for step in steps:
                    self.assertEqual(step.command[0], 'isolated-python')
                    if step.command[1] == '-m':
                        module = importlib.import_module(step.command[2])
                        module.build_parser().parse_args(step.command[3:])
                command = steps[0].command
                data = command[command.index('--data-dir') + 1]
                self.assertEqual(data, '/clip-subj' if method == 'cliptext' and exp == 'utility' else '/raw-subj')
                for step in steps:
                    if '--data-dir' in step.command:
                        self.assertEqual(step.command[step.command.index('--data-dir') + 1], data)
                if method == 'cliptext' and exp == 'utility':
                    self.assertEqual(command[command.index('--original-data-dir') + 1], '/raw-subj')
                if method == 'zo_dpga':
                    self.assertEqual(command[command.index('--method') + 1], 'dpzero' if exp == 'utility' else 'mezo')
                # The GLM backend records its native training report; it does
                # not append the optional unified holdout utility evaluator.
                self.assertIn('training_report', artifacts)
                self.assertNotIn('utility', artifacts)
                self.assertIn('adapter', products)

    def test_privsv_reuses_raw_and_selects_variant(self):
        self.manifest('privsv', products={'adapter': '/shared-raw', 'steering': '/private'})
        steps, _, products = self.build('privsv', 'baseline')
        self.assertEqual(steps[0].command[2], 'privacy_attack.build_steering')
        self.assertIn('mean', steps[0].command)
        self.assertEqual(products['adapter'], '/shared-raw')
        # The shared Raw LoRA is reused as a product for inference; steering
        # construction itself operates on the base model and raw data.
        self.assertEqual(len(steps), 1)
        self.assertIn('/raw-subj', steps[-1].command)
        self.assertNotIn('--adapter', steps[0].command)  # upstream constructs vectors on base

    def test_direct_pair_uses_raw_and_matched_targets(self):
        for method in g.METHODS:
            self.settings['glm_subj']['artifacts'][method] = {
                'protected_adapter': '/protected', 'baseline_adapter': '/baseline',
                'raw_adapter': '/shared', 'protected_steering': '/private', 'baseline_steering': '/mean'}
            steps, artifacts, _ = self.build(method, 'direct_attack')
            for step in steps[:2]:
                module = importlib.import_module(step.command[2])
                module.build_parser().parse_args(step.command[3:])
                self.assertIn('/raw-subj', step.command)
                self.assertNotIn('/clip-subj', step.command)
                if method == 'privsv':
                    self.assertIn('/shared', step.command)
            self.assertIn('--require-matched-targets', steps[-1].command)
            self.assertIn('comparison', artifacts)

    def test_missing_artifacts_and_placeholders(self):
        with self.assertRaisesRegex(ValueError, '先完成'):
            self.build('cliptext', 'direct_attack')
        for exp in g.PLACEHOLDERS:
            with self.assertRaises(NotImplementedError):
                self.build('cliptext', exp)
            with patch.object(g, 'build_steps') as builder, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(g.run(self.settings, 'cliptext', exp), 0)
                builder.assert_not_called()
            self.assertIsNone(g.history(self.settings, 'cliptext', exp))

    def test_history_filters_identity_status_and_recorded_time(self):
        expected = self.manifest(name='match')
        for index, kwargs in enumerate([
            {'method': 'privsv'}, {'experiment': 'baseline'}, {'status': 'failed'},
            {'model': 'llama'}, {'dataset': 'sst2'}, {'backend': 'legacy'}, {'role': 'baseline'},
        ]):
            self.manifest(name=f'wrong-{index}', completed='2027-01-01T00:00:00+00:00', **kwargs)
        self.assertEqual(g.history(self.settings, 'cliptext', 'utility'), expected)
        newer = self.manifest(name='newer', completed='2026-02-01T00:00:00+00:00')
        self.assertEqual(g.history(self.settings, 'cliptext', 'utility'), newer)
        Path(newer['artifacts']['report']).unlink()
        self.assertEqual(g.history(self.settings, 'cliptext', 'utility'), expected)

    def test_dry_run_has_no_manifest_or_execution(self):
        with patch.object(g.subprocess, 'Popen') as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(g.run(self.settings, 'cliptext', 'utility', 'dry'), 0)
        execute.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_completed_and_failed_execution(self):
        steps = [g.Step('fake', ['fake'], self.root)]
        report = self.root / 'artifact.json'
        def execute(*args, **kwargs):
            self.assertEqual(kwargs['env']['PYTHONUNBUFFERED'], '1')
            report.write_text('{}')
            class Process:
                stdout = io.StringIO('epoch 1: loss=0.5\n')
                def wait(self):
                    return 0
            return Process()
        console = io.StringIO()
        with patch.object(g, 'build_steps', return_value=(steps, {'report': str(report)}, {})), \
             patch.object(g, 'doctor'), patch('privacy_tools.bootstrap.check_imports', return_value=(True, [])), \
             patch('privacy_tools.bootstrap.check_cuda', return_value=(True, 'PyTorch 2.8.0+cu128，CUDA 12.8，使用 GPU [3]（共检测到 4 张）')), \
             patch.object(g.subprocess, 'Popen', side_effect=execute), contextlib.redirect_stdout(console):
            self.assertEqual(g.run(self.settings, 'cliptext', 'utility'), 0)
        self.assertIn('epoch 1: loss=0.5', console.getvalue())
        for message in ('[OK] 所需依赖已就绪：isolated-python', '[OK] GPU：PyTorch 2.8.0+cu128',
                        '使用 GPU [3]（共检测到 4 张）', '任务确认', '任务：微调',
                        '模型：ChatGLM3-6B', '数据集：Subj', '运行目录：', '日志文件：'):
            self.assertIn(message, console.getvalue())
        logs = list(self.root.glob('*/run.log'))
        self.assertIn('epoch 1: loss=0.5', logs[0].read_text())
        self.assertIsNotNone(g.history(self.settings, 'cliptext', 'utility'))
        with patch.object(g, 'build_steps', return_value=(steps, {'report': str(report)}, {})), \
             patch.object(g, 'doctor'), patch('privacy_tools.bootstrap.check_imports', return_value=(True, [])), \
             patch('privacy_tools.bootstrap.check_cuda', return_value=(True, 'PyTorch 2.8.0+cu128，CUDA 12.8，使用 GPU [3]（共检测到 4 张）')), \
             patch.object(g.subprocess, 'Popen', side_effect=OSError('failed')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(g.run(self.settings, 'privsv', 'utility'), 1)
        self.assertIsNone(g.history(self.settings, 'privsv', 'utility'))

    def test_selection_and_legacy_isolation(self):
        for method in g.METHODS:
            with patch.object(ui, 'choose_from_menu', side_effect=['glm_subj', 'subj']):
                self.assertEqual(ui._choose_model_and_dataset(self.settings, method_id=method,
                                 experiment_id='utility'), ('ChatGLM3-6B', 'subj'))
            with patch.object(ui, 'choose_from_menu', side_effect=['glm_subj', 'configured']), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ui._choose_model_and_dataset(self.settings, method_id=method,
                                 experiment_id='utility'), (None, None))
        menus = []
        def choose(title, options):
            menus.extend(options)
            return 'configured'
        with patch.object(ui, 'choose_from_menu', side_effect=choose):
            ui._choose_model_and_dataset(self.settings, method_id='kal_dp', experiment_id='utility')
        self.assertNotIn('glm_subj', [x[0] for x in menus])

    def test_gpu_failure_and_declined_confirmation_never_start_training(self):
        for gpu_ready in (False, True):
            console = io.StringIO()
            def confirm():
                self.assertIn('[OK] GPU：', console.getvalue())
                self.assertIn('任务确认', console.getvalue())
                return False
            with patch.object(g, 'doctor'), \
                 patch('privacy_tools.bootstrap.check_imports', return_value=(True, [])), \
                 patch('privacy_tools.bootstrap.check_cuda', return_value=(gpu_ready, 'GPU detail')) as gpu, \
                 patch.object(g.subprocess, 'Popen') as execute, contextlib.redirect_stdout(console):
                code = g.run(self.settings, 'cliptext', 'utility', confirm=confirm)
            self.assertEqual(code, 0 if gpu_ready else 1)
            gpu.assert_called_once_with('isolated-python', '0')
            execute.assert_not_called()
            self.assertEqual(list(self.root.iterdir()), [])

    def test_interactive_routes_before_legacy(self):
        for task, exp in [('finetune', ''), ('evaluation', 'direct_attack')]:
            selection = ui.TaskSelection(task, 'cliptext', exp, 'embedding')
            choices = ['utility', 'dry'] if task == 'finetune' else ['dry']
            with patch.object(ui, '_select_task', return_value=selection), \
                 patch.object(ui, '_choose_model_and_dataset', return_value=('ChatGLM3-6B', 'subj')), \
                 patch.object(ui, 'choose_from_menu', side_effect=choices), \
                 patch.object(g, 'run', return_value=0) as run, \
                 patch.object(ui, '_prepare_assets') as old, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ui.interactive(self.settings), 0)
                self.assertEqual(run.call_args.args, (self.settings, 'cliptext', exp or 'utility', 'dry'))
                self.assertEqual(run.call_args.kwargs['selection'].backend, 'glm_subj')
                old.assert_not_called()

    def test_backend_environment_and_fixed_holdout(self):
        self.assertEqual(environment_plan('glm_subj', 'utility').requirements, g.ROOT / 'requirements.txt')
        from privacy_tools.glm_subj_utility import holdout
        (self.root / 'test.jsonl').write_text('{"text":"sentence","label":1}\n')
        self.assertEqual(holdout(self.root, 0), holdout(self.root, 1))

    def test_cli_routes_backend_without_legacy_execution(self):
        with patch.object(g, 'run', return_value=0) as run, patch.object(ui, 'run_experiment') as old:
            self.assertEqual(ui.main(['run', '--backend', 'glm_subj', '--method', 'zo_dpga',
                                      '--experiment', 'utility', '--dry-run']), 0)
            self.assertEqual(run.call_args.args[1:], ('zo_dpga', 'utility', 'dry'))
            old.assert_not_called()

    def test_doctor_checks_selected_paths_and_external_products(self):
        import sys
        sys.path.insert(0, str(g.ROOT))
        with patch('privacy_attack.common.validate_chatglm_base') as base, \
             patch('privacy_attack.common.load_split') as split, \
             patch('privacy_attack.common.validate_adapter') as adapter:
            steps, _, _ = self.build('cliptext', 'utility')
            g.doctor(self.settings, steps)
            base.assert_called_with(Path('/chatglm'))
            self.assertIn(unittest.mock.call('/clip-subj', 'test'), split.call_args_list)
            adapter.assert_not_called()  # adapter is produced earlier in this run
            self.settings['glm_subj']['artifacts']['cliptext'] = {
                'protected_adapter': '/protected', 'baseline_adapter': '/baseline'}
            steps, _, _ = self.build('cliptext', 'direct_attack')
            g.doctor(self.settings, steps)
            self.assertEqual(adapter.call_count, 2)

    def test_attack_reference_cannot_be_overridden(self):
        self.settings['glm_subj']['artifacts']['cliptext'] = {'protected_adapter': '/a', 'baseline_adapter': '/b'}
        self.settings['glm_subj']['attack'] = {'data_dir': '/wrong'}
        with self.assertRaisesRegex(ValueError, 'reference'):
            self.build('cliptext', 'direct_attack')


if __name__ == '__main__':
    unittest.main()
