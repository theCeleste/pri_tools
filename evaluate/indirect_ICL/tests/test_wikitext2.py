"""CPU checks for WikiText-2 adaptation, including generation-to-metrics flow."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'DPICL'))
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel

from Indirect_attack import combine, neighbour_mia
from Indirect_attack.evaluate_attack_metrics import compute_metrics
import run_exp_sst2_local as generation
from runtime_config import DPICL_DIR
from task_data import generation_full_datum, generation_test_input, load_wikitext2, prepare_wikitext2
import task_data
from test_delivery import TinyTokenizer


class WikiTextTests(unittest.TestCase):
    def test_deduplication_and_split_isolation(self):
        data = DatasetDict({split: Dataset.from_dict({'text': texts}) for split, texts in {
            'train': [' ', 'short', 'a private encyclopedia passage', 'a  private encyclopedia passage'],
            'validation': ['a private encyclopedia passage', 'a held out validation passage'],
            'test': ['a held out validation passage', 'an unseen encyclopedia passage']}.items()})
        cleaned, counts = prepare_wikitext2(data)
        self.assertEqual([len(cleaned[s]) for s in ('train', 'validation', 'test')], [1, 1, 1])
        self.assertEqual(counts['train']['duplicates'], 1)
        self.assertEqual(counts['train']['short_or_blank'], 2)
        self.assertEqual(cleaned['train']['label'], [0])

    def test_no_sentiment_or_class_label_in_wiki_prompts(self):
        row = dict(sentence='an encyclopedia passage', label=0)
        self.assertEqual(generation_full_datum(['text'], row), 'Text: "an encyclopedia passage END"\n')
        self.assertEqual(generation_test_input(['text'], 0), 'Text: "')
        data = [dict(original_text='original passage', synthetic_text='synthetic passage', label=[])]
        for field in ('original_text', 'synthetic_text'):
            prompt = neighbour_mia.build_prompt(data, field, 0, 1, 'wikitext2')
            self.assertNotIn('Label:', prompt)
            self.assertNotIn('Sentiment:', prompt)
            self.assertIn(data[0][field], prompt)

    def test_sst2_format_is_unchanged(self):
        row = dict(sentence='a nice movie', label=1)
        self.assertEqual(generation_full_datum(['negative', 'positive'], row),
                         'Sentiment: "positive"\nText: "a nice movie END"\n')
        self.assertEqual(generation_test_input(['negative', 'positive'], 1),
                         'Sentiment: "positive"\nText: "')

    def test_snapshot_config_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'dataset_config.json').write_text(json.dumps({'config': 'wikitext-2-v1'}))
            with self.assertRaisesRegex(ValueError, 'config'):
                load_wikitext2(path)

    def test_snapshot_export_is_serializable_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            texts = {split: [f'a unique {split} encyclopedia passage']
                     for split in ('train', 'validation', 'test')}
            for split, rows in texts.items():
                (source / f'{split}.json').write_text(json.dumps(rows))
            destination = root / 'snapshot'
            with patch.object(sys, 'argv', ['task_data.py', '--wikitext2-dir', str(source),
                 '--output-dir', str(destination)]), contextlib.redirect_stdout(io.StringIO()):
                task_data.main()
            data, counts = load_wikitext2(destination)
            for split in texts:
                self.assertEqual(list(data[split]['sentence']), texts[split])
                self.assertEqual(counts[split]['kept'], 1)

    def test_wiki_rejects_sentiment_utility(self):
        with self.assertRaisesRegex(ValueError, 'no class labels'):
            generation._main(sigma=.39, task_dataset='wikitext2', num_test=20)

    def test_entrypoint_commands_and_sst2_defaults(self):
        for task in ('sst2', 'wikitext2'):
            result = subprocess.run([sys.executable, '-B', str(DPICL_DIR / 'scripts/run_indirect.py'),
                '--task-dataset', task, '--target-model', 'fm9g', '--run-dir', 'runs/unittest-dry-run',
                '--dry-run'], capture_output=True, text=True, check=True)
            text = result.stdout
            self.assertIn(f'--max-length {2048 if task == "wikitext2" else 512}', text)
            self.assertIn(f'--num-test {0 if task == "wikitext2" else 20}', text)
            self.assertIn('--no-add-noise', text)
            self.assertIn('--proposal-model resources/bert-base-uncased', text)
            self.assertIn('attack_results_other_bert_0_original_text.csv', text)
            self.assertIn(f'attack_results_{task}_bert_0_original_text.csv', text)
            self.assertEqual(text.count('prepare_data:'), int(task == 'wikitext2'))
            self.assertEqual(len(text.splitlines()), 6 if task == 'wikitext2' else 5)

    def test_generation_attack_combine_evaluate_on_cpu(self):
        torch.manual_seed(0)
        tokenizer = TinyTokenizer()
        target = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=128,
            n_embd=16, n_layer=1, n_head=2)).eval()
        proposal = BertForMaskedLM(BertConfig(vocab_size=32, hidden_size=16,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)).eval()

        class Encoding:
            n_vocab = 32
            def encode(self, text):
                return [1]
            def stop_token_ids(self, strings):
                return set()
            def reached_stop(self, ids, strings):
                return False
            def trim_stops(self, ids, strings):
                return 'a synthetic encyclopedia passage'

        lm = SimpleNamespace(encoding=Encoding(), eos_token_ids={2}, input_prefix_tokens=[])
        async def probabilities(*args, **kwargs):
            return {3: 0.}

        old_labels, old_dictionary = generation.labels, generation.label_dict
        self.addCleanup(setattr, generation, 'labels', old_labels)
        self.addCleanup(setattr, generation, 'label_dict', old_dictionary)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            for split, count in [('train', 24), ('validation', 2), ('test', 3)]:
                (source / f'{split}.json').write_text(json.dumps([
                    f'{split} encyclopedia passage number {i}' for i in range(count)]))
            with patch.object(generation, 'select_device', return_value=torch.device('cpu')), \
                 patch.object(generation, 'api_openai_com', return_value=lm), \
                 patch.object(generation, 'next_logprobs', side_effect=probabilities), \
                 patch.object(generation, 'normalized_logprobs_for_chosen_tokens', side_effect=probabilities), \
                 patch.object(generation, 'get_model_response', side_effect=AssertionError('No sentiment utility on WikiText-2')), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                generation._main(sigma=.39, openai_model='fake-model', task_dataset='wikitext2',
                    wikitext2_dir=str(source), output_dir=str(root / 'generated'),
                    num_valid=10, num_test=0, num_private_train=20, num_private_train_splits=10,
                    sample_same_label_prompts=True, subsample_per_token=True,
                    add_noise=False, max_token_per_text=3)
            input_json = root / 'generated/synthetic_text_results.json'
            records = json.loads(input_json.read_text())
            self.assertEqual(len(records), 10)
            self.assertTrue(all(r['task_dataset'] == 'wikitext2' and r['label'] == [] for r in records))
            config = json.loads((root / 'generated/generation_config.json').read_text())
            self.assertEqual(config['effective_sigma'], 0)
            self.assertEqual(config['utility_evaluation'], 'not applicable')

            for field in ('original_text', 'synthetic_text'):
                destination = root / field
                with patch.object(neighbour_mia, 'select_device', return_value=torch.device('cpu')), \
                     patch.object(neighbour_mia, 'load_target', return_value=(tokenizer, target)), \
                     patch.object(neighbour_mia.AutoTokenizer, 'from_pretrained', return_value=tokenizer), \
                     patch.object(neighbour_mia.AutoModelForMaskedLM, 'from_pretrained', return_value=proposal), \
                     contextlib.redirect_stderr(io.StringIO()):
                    for dataset, count in [('other', 10), ('wikitext2', 3)]:
                        neighbour_mia.main(['--dataset', dataset, '--task-dataset', 'wikitext2',
                            '--wikitext2-dir', str(source), '--target-model', 'fake-model',
                            '--input-json', str(input_json), '--prompt-field', field,
                            '--output-dir', str(destination), '--limit', str(count),
                            '--max-length', '96', '--proposal-max-length', '32'])
                combined = destination / 'combined.csv'
                with contextlib.redirect_stdout(io.StringIO()):
                    combine.main(['--member-file', str(destination / f'attack_results_other_bert_0_{field}.csv'),
                        '--nonmember-file', str(destination / f'attack_results_wikitext2_bert_0_{field}.csv'),
                        '--output', str(combined)])
                    metrics = compute_metrics(combined, plot=False, output_dir=destination / 'evaluation')
                frame = pd.read_csv(combined)
                self.assertEqual(frame.label.tolist(), [1] * 10 + [0] * 3)
                np.testing.assert_allclose(frame.diff_score, frame.orig_logprob - frame.mean_neigh_logprob)
                self.assertEqual(metrics['member_count'], 10)
                self.assertEqual(metrics['nonmember_count'], 3)
                for name in ('1%', '0.1%', '0.01%'):
                    self.assertIn(f'TPR@{name}', metrics)
                self.assertFalse({'TPR@100%', 'TPR@50%', 'TPR@10%'} & set(metrics))
                self.assertTrue((destination / 'evaluation/roc.csv').exists())


if __name__ == '__main__':
    unittest.main()
