"""Generate demonstrations, attack both contexts, combine labels and evaluate."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_config import DPICL_DIR, PROJECT_DIR, project_path
from task_data import WIKITEXT_CONFIGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-model', required=True)
    parser.add_argument('--generation-model', help='Defaults to target model; author used base for generation and chat for attack')
    parser.add_argument('--proposal-model', default='resources/bert-base-uncased',
                        help='Bundled BERT weights; no external HF cache required')
    parser.add_argument('--gpu', type=int, choices=range(4), default=0)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float32')
    parser.add_argument('--generation-dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--input-json', help='Reuse existing demonstrations; skip generation')
    parser.add_argument('--sst2-dir', help='Optional local directory containing train.tsv/dev.tsv/test.tsv')
    parser.add_argument('--task-dataset', choices=['sst2', 'wikitext2'], default='sst2')
    parser.add_argument('--wikitext2-dir', help='Optional local train.json/validation.json/test.json directory')
    parser.add_argument('--wikitext2-config', choices=WIKITEXT_CONFIGS, default=WIKITEXT_CONFIGS[0])
    parser.add_argument('--num-nonmembers', type=int, default=500)
    parser.add_argument('--num-demonstrations', type=int, default=10)
    parser.add_argument('--num-test', type=int, help='Classification utility test count; SST-2=20, WikiText-2=0')
    parser.add_argument('--max-token-per-text', type=int, default=100)
    parser.add_argument('--max-length', type=int, help='Target tokenizer limit; SST-2=512, WikiText-2=2048')
    parser.add_argument('--proposal-max-length', type=int, help='Optional proposal tokenizer limit')
    parser.add_argument('--prompt-start', type=int, default=5)
    parser.add_argument('--prompt-stop', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--score-mode', choices=['legacy', 'token-mean'], default='token-mean')
    parser.add_argument('--add-noise', action='store_true', help='Off by default for the requested new experiments')
    parser.add_argument('--sigma', type=float, default=.39)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.num_nonmembers < 1 or args.num_demonstrations < 1:
        parser.error('Sample counts must be positive')
    wiki = args.task_dataset == 'wikitext2'
    if wiki and args.sst2_dir:
        parser.error('--sst2-dir cannot be used for WikiText-2')
    if not wiki and args.wikitext2_dir:
        parser.error('--wikitext2-dir requires --task-dataset wikitext2')
    if args.num_test is None:
        args.num_test = 0 if wiki else 20
    if args.max_length is None:
        args.max_length = 2048 if wiki else 512
    if wiki and args.num_test != 0:
        parser.error('WikiText-2 has no classification labels; use --num-test 0')
    if args.num_test < 0 or args.max_length < 2:
        parser.error('--num-test must be nonnegative and --max-length must be at least 2')
    if args.proposal_max_length is not None and args.proposal_max_length < 1:
        parser.error('--proposal-max-length must be positive')
    if not 0 <= args.prompt_start < args.prompt_stop <= args.num_demonstrations:
        parser.error('Prompt range must be nonempty and within --num-demonstrations')
    root = project_path(args.run_dir)
    generated = root / 'generated'
    input_json = project_path(args.input_json) if args.input_json else generated / 'synthetic_text_results.json'
    stages = []
    def stage(name, script, options):
        stages.append((name, [sys.executable, '-u', str(DPICL_DIR / script), *map(str, options)]))
    trust = ['--trust-remote-code'] if args.trust_remote_code else []
    snapshot = root / 'dataset'
    wiki_options = ['--task-dataset', 'wikitext2', '--wikitext2-dir', snapshot,
                    '--wikitext2-config', args.wikitext2_config] if wiki else []
    if wiki:
        options = ['--wikitext2-config', args.wikitext2_config, '--output-dir', snapshot]
        if args.wikitext2_dir:
            options += ['--wikitext2-dir', args.wikitext2_dir]
        stage('prepare_data', 'task_data.py', options)
    if not args.input_json:
        options = ['--sigma', args.sigma, '--openai-model', args.generation_model or args.target_model,
                   '--num-private-train', 20, '--set-num-public-train', 0,
                   '--num-private-train-splits', 10, '--num-valid', args.num_demonstrations,
                   '--num-test', args.num_test, '--max-token-per-text', args.max_token_per_text,
                   '--sample-same-label-prompts', '--subsample-per-token',
                   '--synth-seed', args.seed, '--eval-seed', args.seed, '--gpu', args.gpu,
                   '--dtype', args.generation_dtype, '--output-dir', generated,
                   '--add-noise' if args.add_noise else '--no-add-noise', *wiki_options, *trust]
        if args.sst2_dir:
            options += ['--sst2-dir', args.sst2_dir]
        stage('generation', 'run_exp_sst2_local.py', options)
    for field in ['original_text', 'synthetic_text']:
        destination = root / field
        nonmember_dataset = args.task_dataset
        for dataset in ['other', nonmember_dataset]:
            options = ['--dataset', dataset, '--proc-id', 0, '--model', 'bert',
                       '--target-model', args.target_model, '--gpu', args.gpu, '--dtype', args.dtype,
                       '--input-json', input_json, '--prompt-field', field,
                       '--prompt-start', args.prompt_start, '--prompt-stop', args.prompt_stop,
                       '--output-dir', destination, '--seed', args.seed, '--score-mode', args.score_mode,
                       '--max-length', args.max_length,
                       '--proposal-model', args.proposal_model,
                       '--batch-size', max(args.num_demonstrations, args.num_nonmembers),
                       '--limit', args.num_nonmembers if dataset != 'other' else args.num_demonstrations,
                       *wiki_options, *trust]
            if args.proposal_max_length is not None:
                options += ['--proposal-max-length', args.proposal_max_length]
            if args.sst2_dir and dataset == 'sst2':
                options += ['--nonmember-file', project_path(args.sst2_dir) / 'test.tsv']
            stage(f'{field}_{dataset}', 'Indirect_attack/neighbour_mia.py', options)
        combined = destination / 'combined.csv'
        stage(f'{field}_combine', 'Indirect_attack/combine.py', [
            '--member-file', destination / f'attack_results_other_bert_0_{field}.csv',
            '--nonmember-file', destination / f'attack_results_{nonmember_dataset}_bert_0_{field}.csv', '--output', combined])
        stage(f'{field}_evaluate', 'Indirect_attack/evaluate_attack_metrics.py', [
            '--file', combined, '--output-dir', destination / 'evaluation'])
    if args.dry_run:
        for name, command in stages:
            print(name + ': ' + shlex.join(command))
        return
    if root.exists():
        raise FileExistsError(f'Choose a new --run-dir; refusing to overwrite {root}')
    root.mkdir(parents=True)
    (root / 'run_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    timings = []
    for name, command in stages:
        start = time.monotonic()
        print(f'START {name}; log: {root / (name + ".log")}', flush=True)
        with (root / (name + '.log')).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=PROJECT_DIR, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        timings.append(dict(stage=name, seconds=time.monotonic() - start, returncode=returncode))
        (root / 'stage_timings.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
        if returncode:
            raise RuntimeError(f'{name} failed; inspect its log; subsequent stages were not started')
        print(f'DONE {name}: {timings[-1]["seconds"]:.1f}s', flush=True)
    print(f'All stages completed: {root}', flush=True)


if __name__ == '__main__':
    main()
