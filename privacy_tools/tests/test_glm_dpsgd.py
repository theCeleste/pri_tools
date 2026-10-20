"""CPU contracts for ChatGLM DP-SGD routing and result isolation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from privacy_tools import glm_subj as g, unified as ui
from privacy_tools.bootstrap import environment_plan


class GlmDpsgdTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = {'python': {'glm_subj_dpsgd': 'dp-python'},
                         'common': {'gpu': '3'}, 'glm_subj': {
                             'base_model': str(self.root / 'model'),
                             'output_root': str(self.root / 'runs'),
                             'dpsgd': {'epochs': 2}}}

    def test_all_methods_use_dpsgd_with_isolated_outputs(self):
        for method in g.METHODS:
            steps, artifacts, products = g.build_steps(
                self.settings, self.root / method, method, 'performance_baseline')
            step = steps[0]
            self.assertEqual(step.command[:2], ['dp-python', str(g.DPSGD_ROOT / 'dpsgd_chatglm3_subj.py')])
            self.assertEqual(step.command[step.command.index('--epochs') + 1], '2')
            self.assertEqual(step.command[step.command.index('--output_dir') + 1], str(self.root / method))
            self.assertEqual(step.env['CUDA_VISIBLE_DEVICES'], '3')
            self.assertEqual(artifacts['training_report'], str(self.root / method / 'training_report.json'))
            self.assertEqual(g.identity(method, 'performance_baseline')['role'], 'baseline')
            self.assertNotIn('--data-dir', step.command)

    def test_invalid_settings_rejected(self):
        for values in ({'epochs': 0}, {'epochs': 1.5}, {'learning_rate': float('nan')},
                       {'model_name': '/override'}, {'output_dir': '/override'}):
            self.settings['glm_subj']['dpsgd'] = values
            with self.assertRaises(ValueError):
                g.build_steps(self.settings, self.root, 'cliptext', 'performance_baseline')
        self.settings['glm_subj']['dpsgd'] = {}
        self.settings['common']['gpu'] = '0,1'
        with self.assertRaisesRegex(ValueError, '单卡'):
            g.build_steps(self.settings, self.root, 'cliptext', 'performance_baseline')

    def test_dry_and_cli_do_not_execute_or_create_outputs(self):
        with patch.object(g.subprocess, 'Popen') as execute, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(g.run(self.settings, 'privsv', 'performance_baseline', 'dry'), 0)
        execute.assert_not_called()
        self.assertIn('dpsgd_chatglm3_subj.py', out.getvalue())
        self.assertFalse((self.root / 'runs').exists())
        with patch.object(g, 'run', return_value=0) as run:
            self.assertEqual(ui.main(['run', '--backend', 'glm_subj', '--method', 'cliptext',
                                     '--experiment', 'performance_baseline', '--dry-run']), 0)
        self.assertEqual(run.call_args.args[1:], ('cliptext', 'performance_baseline', 'dry'))

    def test_environment_and_model_validation(self):
        self.assertEqual(environment_plan('glm_subj_dpsgd', 'performance_baseline').requirements,
                         g.DPSGD_ROOT / 'requirements.txt')
        steps, _, _ = g.build_steps(self.settings, self.root, 'cliptext', 'performance_baseline')
        model = self.root / 'model'
        model.mkdir()
        for config, valid in [({'model_type': 'chatglm'}, False),
                              ({'model_type': 'chatglm', 'auto_map': {
                                  'AutoModelForSequenceClassification': 'modeling_chatglm.ChatGLMForSequenceClassification'}}, True)]:
            (model / 'config.json').write_text(json.dumps(config))
            if valid:
                g.doctor(self.settings, steps)
            else:
                with self.assertRaises(ValueError):
                    g.doctor(self.settings, steps)

    def test_execution_history_and_missing_report(self):
        def execute(command, **kwargs):
            directory = Path(command[command.index('--output_dir') + 1])
            (directory / 'training_report.json').write_text('{"epsilon": 7.9}')
            class Process:
                stdout = io.StringIO('DP-SGD training\n')
                def wait(self):
                    return 0
            return Process()
        with patch.object(g, 'doctor'), \
             patch('privacy_tools.bootstrap.check_imports', return_value=(True, [])) as imports, \
             patch('privacy_tools.bootstrap.check_cuda', return_value=(True, 'GPU 3')), \
             patch.object(g.subprocess, 'Popen', side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(g.run(self.settings, 'cliptext', 'performance_baseline'), 0)
        self.assertEqual(imports.call_args.args[0], 'dp-python')
        result = g.history(self.settings, 'cliptext', 'performance_baseline')
        self.assertEqual(result['status'], 'success')
        self.assertIsNone(g.history(self.settings, 'cliptext', 'baseline'))
        self.assertIsNone(g.history(self.settings, 'privsv', 'performance_baseline'))
        with patch.object(g.subprocess, 'Popen') as process, \
             patch('privacy_tools.bootstrap.check_cuda') as gpu, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(g.run(self.settings, 'cliptext', 'performance_baseline', 'history'), 0)
        process.assert_not_called()
        gpu.assert_not_called()
        Path(result['artifacts']['training_report']).unlink()
        self.assertIsNone(g.history(self.settings, 'cliptext', 'performance_baseline'))


if __name__ == '__main__':
    unittest.main()
