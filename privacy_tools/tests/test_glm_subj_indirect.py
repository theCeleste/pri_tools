"""CPU contracts for SUBJ paired indirect evaluation; no model is loaded."""
import contextlib
import csv
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from privacy_tools import glm_subj as g, glm_subj_indirect as i, unified as ui
from privacy_tools.bootstrap import environment_plan


class IndirectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = {'python': {'glm_subj': sys.executable}, 'glm_subj': {
            'base_model': '/home/zjs/llm/chatglm3-6b', 'output_root': str(self.root)}}

    def write_json(self, file, value):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(value), encoding='utf-8')

    def test_all_commands_parse_and_forward_native_stages(self):
        for method, (native, baseline, protected) in i.METHODS.items():
            steps, artifacts, products = g.build_steps(self.settings, self.root / method, method, 'indirect_attack')
            command = steps[0].command
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(steps[0].cwd, i.ROOT)
            self.assertEqual(products, {})
            self.assertEqual(g.identity(method, 'indirect_attack')['role'], 'paired')
            self.assertEqual(set(artifacts), {'baseline_metrics', 'protected_metrics', 'indirect_summary'})
            output = subprocess.check_output(command + ['--dry-run'], cwd=steps[0].cwd,
                                             env={**os.environ, **steps[0].env}, text=True)
            self.assertIn(str(self.root / method / 'indirect'), output)
            if method == 'privsv':
                self.assertIn('--subset-seed 60', output)
                self.assertIn('--seed 43', output)
                self.assertNotIn('--adapter', output)
            else:
                for variant in (baseline, protected):
                    display_variant = 'zo_dpga' if variant == 'dpzero' else variant
                    self.assertIn(str(i.ROOT / native / 'weights' / display_variant), output)
                self.assertIn('--max-length 14' if method == 'zo_dpga' else '--max-length 512', output)
                self.assertIn('--loss-prefix-tokens 8' if method == 'zo_dpga' else '--loss-prefix-tokens 0', output)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_preview_and_history_never_check_gpu_or_execute(self):
        for method in i.METHODS:
            with patch.object(g.subprocess, 'Popen') as execute, \
                 patch('privacy_tools.bootstrap.check_cuda') as gpu, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(g.run(self.settings, method, 'indirect_attack', 'dry'), 0)
                self.assertEqual(g.run(self.settings, method, 'indirect_attack', 'history'), 0)
                execute.assert_not_called()
                gpu.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_selection_routes_all_methods_and_cli_before_legacy(self):
        for method in i.METHODS:
            selection = ui.TaskSelection('evaluation', method, 'indirect_attack', 'embedding')
            with patch.object(ui, '_select_task', return_value=selection), \
                 patch.object(ui, '_choose_model_and_dataset', return_value=('ChatGLM3-6B', 'subj')), \
                 patch.object(ui, 'choose_from_menu', return_value='dry'), \
                 patch.object(g, 'run', return_value=0) as run, \
                 patch.object(ui, '_prepare_assets') as legacy, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ui.interactive(self.settings), 0)
                self.assertEqual(run.call_args.args[1:], (method, 'indirect_attack', 'dry'))
                legacy.assert_not_called()
            with patch.object(g, 'run', return_value=0) as run, patch.object(ui, 'run_experiment') as legacy:
                self.assertEqual(ui.main(['run', '--backend', 'glm_subj', '--method', method,
                                         '--experiment', 'indirect_attack', '--dry-run']), 0)
                self.assertEqual(run.call_args.args[1:], (method, 'indirect_attack', 'dry'))
                legacy.assert_not_called()

    def test_invalid_configs_and_external_proof_required(self):
        for gpu in ('-1', '0,1', '', 'abc', True):
            self.settings['common'] = {'gpu': gpu}
            with self.assertRaises(ValueError):
                g.build_steps(self.settings, self.root, 'cliptext', 'indirect_attack')
        self.settings.pop('common')
        for cfg in ([], {'unknown': 1}, {'resource_source': 'unknown'},
                    {'embedding': {'protected_adapter': '/x'}},
                    {'resource_source': 'external'},
                    {'resource_source': 'external', 'embedding': {
                        'protected_adapter': '/a', 'baseline_adapter': '/b', 'splits': '/s'}}):
            self.settings['glm_subj']['indirect'] = cfg
            with self.assertRaises(ValueError):
                i.resources(self.settings, 'cliptext')

    def test_sample_validation_and_selected_resource_checks(self):
        for method in i.METHODS:
            info = i.resources(self.settings, method)
            sample = i.samples(info)
            self.assertEqual(len(sample['member']), 1000 if method == 'privsv' else 100)
        with self.assertRaisesRegex(ValueError, '交集'):
            i.validate_groups({'member': [{'text': 'Same'}], 'nonmember': [{'text': 'same'}]})
        with self.assertRaisesRegex(ValueError, '重复'):
            i.validate_groups({'member': [{'text': 'a'}, {'text': ' A '}]})
        with patch.object(i, 'required_files', return_value=[self.root / 'missing']):
            with self.assertRaisesRegex(ValueError, '资源缺失'):
                i.doctor(self.settings, 'cliptext')
        plan = environment_plan(g.runtime_profile('indirect_attack'), 'indirect_attack')
        self.assertEqual(plan.requirements, i.ROOT / 'requirements.txt')
        self.assertIn('pandas', plan.imports)
        self.assertNotIn('opacus', plan.imports)
        self.assertEqual(i.python(self.settings), sys.executable)

    def test_external_proof_binds_weights_and_member_texts(self):
        info = {'provenance': str(self.root / 'proof.json')}
        proof = {}
        for role in ('baseline', 'protected'):
            folder = self.root / role
            folder.mkdir()
            (folder / 'adapter_model.safetensors').write_text(role)
            info[role + '_adapter'] = str(folder)
            proof[role] = {'adapter_sha256': i.digest(folder / 'adapter_model.safetensors'),
                           'original_training_texts': ['member text']}
        self.write_json(Path(info['provenance']), proof)
        groups = {'member': [{'text': 'member text'}], 'nonmember': [{'text': 'nonmember text'}]}
        i.validate_provenance(info, groups)
        groups['member'][0]['text'] = 'unknown'
        with self.assertRaisesRegex(ValueError, '不属于'):
            i.validate_provenance(info, groups)
        groups['member'][0]['text'] = 'member text'
        groups['nonmember'][0]['text'] = 'member text'
        with self.assertRaisesRegex(ValueError, 'nonmember'):
            i.validate_provenance(info, groups)
        proof['baseline']['adapter_sha256'] = 'wrong'
        self.write_json(Path(info['provenance']), proof)
        with self.assertRaisesRegex(ValueError, '指纹'):
            i.validate_provenance(info, groups)

    def test_neighbor_cache_requires_text_id_and_generation_match(self):
        info = i.resources(self.settings, 'cliptext')
        info['neighbors'] = str(self.root / 'neighbors.jsonl')
        groups = {'member': [{'source_index': 1, 'text': 'member'}],
                  'nonmember': [{'source_index': 2, 'text': 'nonmember'}]}
        cache = [{'membership': k, 'sample_index': 0, 'text': v[0]['text'],
                  'neighbors': ['neighbor'] * 10} for k, v in groups.items()]
        def save():
            Path(info['neighbors']).write_text(''.join(json.dumps(r)+'\n' for r in cache))
        save()
        self.write_json(Path(info['neighbors'] + '.config.json'), dict(seed=42, proposal_max_length=512,
                        neighbor_top_k=6, candidate_neighbors=50, num_neighbors=10, dropout=.7,
                        proposal_model=info['proposal_model']))
        i.validate_neighbors(info, groups)
        cache[0]['text'] = 'wrong'
        save()
        with self.assertRaisesRegex(ValueError, '正文'):
            i.validate_neighbors(info, groups)
        self.settings['glm_subj']['indirect'] = {'neighbors': info['neighbors']}
        steps, _, _ = i.build_steps(self.settings, self.root, 'cliptext')
        self.assertEqual(steps[0].command[steps[0].command.index('--mode') + 1], 'attack')

    def create_scores(self, method, directory):
        info = i.resources(self.settings, method)
        groups = i.samples(info)
        if method == 'privsv':
            groups['nonmember'] = groups['nonmember'][:1000]
        fields = ['membership', 'sample_index', 'source_index', 'text', 'label', 'num_neighbors',
                  'score', 'original_score', 'neighbor_mean']
        for role, variant in info['variants'].items():
            folder = directory / 'indirect' / variant
            folder.mkdir(parents=True)
            with (folder / 'scores.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for membership, group in groups.items():
                    for index, row in enumerate(group):
                        score = .8 if membership == 'member' else .2
                        writer.writerow(dict(membership=membership, sample_index=index, source_index=row['source_index'],
                                             text=row['text'], label=row.get('label', 1), num_neighbors=10,
                                             score=score, original_score=score, neighbor_mean=0))
            subprocess.run([sys.executable, str(i.ROOT / 'evaluate.py'), '--scores', str(folder / 'scores.csv'),
                            '--output', str(folder / 'evaluation'), '--members', str(len(groups['member'])),
                            '--nonmembers', '1000'], check=True, capture_output=True)
            self.write_json(folder / 'run_config.json', dict(neighbors=str((directory / 'indirect/neighbors.jsonl').resolve()),
                            adapter=info.get(role + '_adapter')))
        if method == 'privsv':
            self.write_json(directory / 'indirect/selection.json', {'subset_seed': 60,
                **{k + '_source_indices': [r['source_index'] for r in group] for k, group in groups.items()}})
        return info

    def test_real_native_metrics_and_paired_summary_all_methods(self):
        for method in i.METHODS:
            directory = self.root / method
            self.create_scores(method, directory)
            summary = i.finalize(self.settings, method, directory)
            self.assertEqual(summary['results']['baseline']['auc'], 1.)
            self.assertEqual(summary['protected_minus_baseline_auc'], 0.)
            metrics = directory / 'indirect' / summary['variants']['protected'] / 'evaluation/metrics.json'
            invalid = i.read_json(metrics)
            invalid['metrics'][0]['tp'] = -1
            self.write_json(metrics, invalid)
            summary = i.finalize(self.settings, method, directory)
            self.assertTrue(any(
                warning['role'] == 'protected' and 'tp' in warning['mismatches']
                for warning in summary['metric_validation_warnings']
            ))

    def test_rejects_mismatched_pairs_partial_and_nonfinite_scores(self):
        directory = self.root / 'scores'
        info = self.create_scores('cliptext', directory)
        file = directory / 'indirect' / info['variants']['protected'] / 'scores.csv'
        original = file.read_text()
        file.write_text(original.replace(',0.8,', ',nan,', 1))
        with self.assertRaisesRegex(ValueError, 'NaN'):
            i.finalize(self.settings, 'cliptext', directory)
        file.write_text(original.replace('harry is watching', 'wrong text', 1))
        with self.assertRaisesRegex(ValueError, '来源 ID'):
            i.finalize(self.settings, 'cliptext', directory)
        file.write_text('\n'.join(original.splitlines()[:-1]) + '\n')
        with self.assertRaisesRegex(ValueError, '不完整'):
            i.finalize(self.settings, 'cliptext', directory)

    def test_history_filters_schema_source_and_keeps_resources_optional(self):
        directory = self.root / 'history'
        report = directory / 'summary.json'
        self.write_json(report, {})
        manifest = {**g.identity('cliptext', 'indirect_attack'), **i.metadata(self.settings, 'cliptext'),
                    'status': 'success', 'completed_at': '2026-10-21T00:00:00+00:00',
                    'artifacts': {'summary': str(report)}}
        file = directory / 'run_manifest.json'
        self.write_json(file, manifest)
        with patch.object(i, 'doctor', side_effect=AssertionError('history must be read only')):
            self.assertIsNotNone(g.history(self.settings, 'cliptext', 'indirect_attack'))
        for key, value in [('schema_version', 99), ('resource_source', 'external'), ('status', 'failed'),
                           ('evaluation_backend', 'old'), ('method', 'privsv')]:
            self.write_json(file, {**manifest, key: value})
            self.assertIsNone(g.history(self.settings, 'cliptext', 'indirect_attack'))

    def test_execution_success_and_bad_metrics_failure_are_manifested(self):
        for good in (True, False):
            def execute(command, **kwargs):
                directory = Path(command[command.index('--output') + 1]).parent
                self.create_scores('cliptext', directory)
                if not good:
                    variant = i.METHODS['cliptext'][2]
                    (directory / 'indirect' / variant / 'scores.csv').write_text('membership,score\n')
                class Process:
                    stdout = io.StringIO('native test subprocess\n')
                    def wait(self):
                        return 0
                return Process()
            # create_scores invokes the real native metric subprocess before returning the fake model process.
            original_popen = subprocess.Popen
            def dispatch(command, **kwargs):
                return original_popen(command, **kwargs) if 'evaluate.py' in str(command[1]) else execute(command, **kwargs)
            with patch.object(g, 'doctor'), patch.object(i, 'metadata', side_effect=lambda s,m,**k: i.resources(s,m)), \
                 patch('privacy_tools.bootstrap.check_imports', return_value=(True, [])), \
                 patch('privacy_tools.bootstrap.check_cuda', return_value=(True, 'test GPU')), \
                 patch.object(g.subprocess, 'Popen', side_effect=dispatch), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(g.run(self.settings, 'cliptext', 'indirect_attack'), 0 if good else 1)
        manifests = [i.read_json(p) for p in self.root.glob('*/run_manifest.json')]
        self.assertEqual(sorted(m['status'] for m in manifests), ['failed', 'success'])
        self.assertIsNotNone(g.history(self.settings, 'cliptext', 'indirect_attack'))

    def test_original_delivery_contracts(self):
        # Original tests only require tmp_path; run them without installing a pytest runtime.
        spec = importlib.util.spec_from_file_location('subj_delivery', i.ROOT / 'tests/test_delivery.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith('test_'):
                with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                    function(Path(directory)) if inspect.signature(function).parameters else function()


if __name__ == '__main__':
    unittest.main()
