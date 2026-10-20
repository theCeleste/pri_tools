"""Isolated ChatGLM3/Subj command routing and manifest-backed history."""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import (PROJECT_ROOT, Step, configured_python, get_nested, method_env,
                   printable_command, project_path, safe_console_write)

ROOT = PROJECT_ROOT / 'Glm-subj'
METHODS = {'cliptext', 'privsv', 'zo_dpga'}
PLACEHOLDERS = {'performance_baseline', 'indirect_attack'}


def config(settings, key, default=None):
    return get_nested(settings, f'glm_subj.{key}', default)


def path(settings, key, default):
    return project_path(config(settings, key, default)).resolve()


def identity(method, experiment):
    if method not in METHODS:
        raise ValueError('Glm-subj 仅支持 ClipText、PrivSV、ZO-DPGA')
    return dict(backend='glm_subj', method=method, model='chatglm3-6b', dataset='subj',
                experiment=experiment, role='paired' if experiment == 'direct_attack' else
                'baseline' if experiment == 'baseline' else 'protected')


def output_root(settings):
    return path(settings, 'output_root', 'Glm-subj/outputs')


def history(settings, method, experiment):
    """Select by manifest identity and recorded completion time, never directory mtime."""
    if experiment in PLACEHOLDERS:
        return None
    expected = identity(method, experiment)
    candidates = []
    for manifest in output_root(settings).glob('*/run_manifest.json'):
        try:
            data = json.loads(manifest.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                continue
            if data.get('status') != 'success' or any(data.get(k) != v for k, v in expected.items()):
                continue
            artifacts = data.get('artifacts', {})
            if not isinstance(artifacts, dict) or not artifacts or not all(Path(p).is_file() for p in artifacts.values()):
                continue
            completed = datetime.fromisoformat(data['completed_at'])
            if completed.tzinfo is None:
                continue
            candidates.append((completed, str(manifest), data))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None


def artifact_pair(settings, method):
    """Explicit artifacts win; otherwise resolve successful training manifests."""
    result = {}
    for role, experiment in [('protected', 'utility'), ('baseline', 'baseline')]:
        previous = history(settings, method, experiment)
        produced = previous.get('products', {}) if previous else {}
        for kind in ('adapter', 'steering'):
            configured = config(settings, f'artifacts.{method}.{role}_{kind}')
            value = str(project_path(configured).resolve()) if configured else produced.get(kind)
            if value:
                result[f'{role}_{kind}'] = Path(value)
    if method == 'privsv':
        raw = config(settings, 'artifacts.privsv.raw_adapter')
        if raw:
            adapter = project_path(raw).resolve()
        else:
            adapter = result.get('protected_adapter') or result.get('baseline_adapter')
        if adapter:
            result.update(protected_adapter=adapter, baseline_adapter=adapter)
    return result


def build_steps(settings, run_dir: Path, method: str, experiment: str):
    identity(method, experiment)
    if experiment in PLACEHOLDERS:
        raise NotImplementedError('ChatGLM3-6B + Subj：DP-SGD / 间接评估尚未接入')
    if experiment not in {'utility', 'baseline', 'direct_attack'}:
        raise ValueError(f'Glm-subj 不支持实验 {experiment}')
    run_dir = run_dir.resolve()
    base = path(settings, 'base_model', get_nested(settings, 'common.chatglm3_6b', 'models/chatglm3-6b'))
    raw = path(settings, 'raw_data', 'Glm-subj/data/subj_raw_prepared/subj/none_a3c561096c597e10')
    embedding = path(settings, 'embedding_data', 'Glm-subj/data/subj_embedding_protected')
    seed = config(settings, 'seed', 42)
    shared = ['--base-model', base, '--seed', seed, '--dtype', config(settings, 'dtype', 'bfloat16')]
    steps = []
    artifacts, products = {}, {}

    def module(name, args):
        steps.append(Step(name, [configured_python(settings, 'glm_subj'), '-m',
                               f'privacy_attack.{name}', *map(str, args)], ROOT,
                          method_env(settings, ROOT, PROJECT_ROOT)))

    if experiment == 'direct_attack':
        pair = artifact_pair(settings, method)
        for role in ('protected', 'baseline'):
            adapter = pair.get(f'{role}_adapter')
            if adapter is None:
                raise ValueError(f'缺少 {method} {role} adapter；请先完成对应训练或配置 glm_subj.artifacts')
            args = [*shared, '--data-dir', raw, '--adapter', adapter,
                    '--prompt-mode', 'raw', '--output-dir', run_dir / 'attacks' / role / 'prefix']
            if method == 'privsv':
                steering = pair.get(f'{role}_steering')
                if steering is None:
                    raise ValueError(f'缺少 {role} steering；请先完成对应构建')
                args += ['--steering', steering]
            attack = config(settings, 'attack', {})
            allowed_attack = {'targets_per_split', 'samples_per_prefix', 'prefix_tokens',
                              'min_suffix_tokens', 'max_new_tokens', 'temperature_start',
                              'temperature_end', 'top_p', 'top_k', 'fuzzy_threshold', 'minimum_fuzzy_tokens'}
            if set(attack) - allowed_attack:
                raise ValueError('攻击配置不能覆盖模型、reference、seed 或输出路径')
            for key, value in attack.items():
                args += ['--' + key.replace('_', '-'), value]
            module('prefix_attack', args)
            artifacts[role + '_summary'] = str(run_dir / 'attacks' / role / 'prefix' / 'summary.json')
        comparison = run_dir / 'attacks' / 'comparison.json'
        module('compare', ['--result', 'baseline', 'source-original', artifacts['baseline_summary'],
                           '--result', 'protected', 'source-original', artifacts['protected_summary'],
                           '--require-matched-targets', '--output-json', comparison,
                           '--output-csv', comparison.with_suffix('.csv')])
        artifacts['comparison'] = str(comparison)
    else:
        protected = experiment == 'utility'
        model_dir = run_dir / 'models' / ('protected' if protected else 'baseline')
        adapter = model_dir / 'adapter'
        data = embedding if method == 'cliptext' and protected else raw
        steering = None
        if method == 'privsv':
            pair = artifact_pair(settings, method)
            adapter = pair.get('protected_adapter') or pair.get('baseline_adapter') or run_dir / 'models' / 'raw'
            # A configured or previously trained raw adapter is shared by both variants.
            if adapter == run_dir / 'models' / 'raw':
                module('train_lora', [*shared, '--data-dir', raw, '--output-dir', adapter,
                                      *_options(config(settings, 'lora', {}))])
            variant = 'private' if protected else 'mean'
            module('build_steering', [*shared, '--data-dir', raw, '--output-dir', model_dir / 'steering',
                                      '--variants', variant, *_options(config(settings, 'steering', {}))])
            steering = model_dir / 'steering' / variant
            artifacts['build_report'] = str(model_dir / 'steering' / 'build_report.json')
            products['steering'] = str(steering)
        elif method == 'cliptext':
            module('train_lora', [*shared, '--data-dir', data, '--output-dir', adapter,
                                  *(['--original-data-dir', raw] if protected else []),
                                  *_options(config(settings, 'lora', {}))])
        else:
            module('train_zo', [*shared, '--method', 'dpzero' if protected else 'mezo',
                                '--data-dir', raw, '--output-dir', adapter,
                                *_options(config(settings, 'zo', {}))])
        artifacts['training_report'] = str(adapter / 'training_report.json')
        products['adapter'] = str(adapter)
        # GLM 微调完成后不再追加统一 Subj holdout 分类效用评估。
        # utility = model_dir / 'utility.json'
        # args = [*shared, '--data-dir', raw, '--adapter', adapter, '--output', utility,
        #         '--max-samples', config(settings, 'utility_samples', 0)]
        # if steering:
        #     args += ['--steering', steering]
        # steps.append(Step('统一 Subj holdout 分类效用', [configured_python(settings, 'glm_subj'),
        #                   str(PROJECT_ROOT / 'privacy_tools' / 'glm_subj_utility.py'), *map(str, args)],
        #                   ROOT, method_env(settings, ROOT, PROJECT_ROOT)))
        # artifacts['utility'] = str(utility)
    return steps, artifacts, products


def _options(values):
    # Reserved paths/identity cannot be overridden through arbitrary command flags.
    allowed = {'rank', 'lora_alpha', 'lora_dropout', 'epochs', 'batch_size', 'gradient_accumulation',
               'learning_rate', 'max_length', 'max_train_samples', 'eval_samples', 'max_steps',
               'zo_epsilon', 'clip_threshold', 'dp_epsilon', 'dp_delta', 'eval_steps',
               'construction_samples', 'clip', 'noise_multiplier', 'multiplier'}
    if set(values) - allowed:
        raise ValueError(f'不支持的 Glm-subj 参数: {sorted(set(values) - allowed)}')
    return [item for key, value in values.items() for item in ('--' + key.replace('_', '-'), value)]


def doctor(settings, steps):
    """Check the selected backend's actual model, data and external products."""
    import sys
    sys.path.insert(0, str(ROOT))
    from privacy_attack.common import load_split, validate_adapter, validate_chatglm_base
    from privacy_attack.steering import validate_steering
    generated = {Path(s.command[s.command.index('--output-dir') + 1]) for s in steps
                 if '--output-dir' in s.command and s.command[3] != 'privacy_attack.prefix_attack'}
    for step in steps:
        args = step.command
        for flag, validator in [('--base-model', validate_chatglm_base), ('--adapter', validate_adapter),
                                 ('--steering', validate_steering)]:
            if flag in args:
                target = Path(args[args.index(flag) + 1])
                if not any(target == p or p in target.parents for p in generated):
                    validator(target)
        if '--data-dir' in args:
            for split in ('train', 'validation', 'test'):
                load_split(args[args.index('--data-dir') + 1], split)


def task_summary(settings, method, experiment, mode, selection=None):
    titles = {'cliptext': 'ClipText', 'privsv': 'PrivSV', 'zo_dpga': 'ZO-DPGA'}
    evaluation = experiment in {'direct_attack', 'indirect_attack'}
    print('\n---------------- 任务确认 ----------------')
    print(f"任务：{'评估' if evaluation else '微调'}")
    print(f'方法：{titles[method]}（{method}）')
    if evaluation:
        print(f"评估类型：{'直接评估' if experiment == 'direct_attack' else '间接评估'}")
        if selection is not None:
            print(f"微调方式：{'显式微调' if selection.finetune_type == 'explicit' else '隐式微调'}")
    else:
        baseline = {'cliptext': 'PEFT微调', 'privsv': 'Prompt微调',
                    'zo_dpga': 'PEFT微调'}[method]
        comparison = ('是，运行 DP-SGD' if experiment == 'performance_baseline' else
                      f'是，运行 {baseline}' if experiment == 'baseline' else '否')
        print(f'是否运行对比方法：{comparison}')
    print('模型：ChatGLM3-6B')
    print('数据集：Subj')
    mode_title = {'run': '立即执行', 'history': '查看最近历史结果', 'dry': '预览命令（不执行）'}[mode]
    print(f'模式：{mode_title}')


def run(settings, method, experiment, mode='run', *, confirm=None, selection=None):
    identity(method, experiment)
    if experiment in PLACEHOLDERS:
        task_summary(settings, method, experiment, mode, selection)
        print('ChatGLM3-6B + Subj：尚未接入，无可用结果。')
        return 0
    if mode == 'history':
        print(f"  查看最近历史结果使用默认配置的 Python环境：{configured_python(settings, 'glm_subj')}")
        print('  查看最近历史结果不会安装依赖或检查 GPU。')
        task_summary(settings, method, experiment, mode, selection)
        previous = history(settings, method, experiment)
        if not previous:
            print('Glm-subj 无匹配的成功历史结果；请先完成对应训练/评估。')
            return 0
        if previous.get('run_directory'):
            print(f"\n运行目录：{previous['run_directory']}")
        print(json.dumps(previous, ensure_ascii=False, indent=2))
        for name, file in previous['artifacts'].items():
            print(f'\n{name}: {file}')
            print(Path(file).read_text(encoding='utf-8'))
        return 0
    run_dir = output_root(settings) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    try:
        steps, artifacts, products = build_steps(settings, run_dir, method, experiment)
        if mode == 'dry':
            print(f"  命令预览使用默认配置的 Python环境：{configured_python(settings, 'glm_subj')}")
            print('  命令预览不会检查 GPU 或执行实验。')
            task_summary(settings, method, experiment, mode, selection)
            print(f'\n运行目录（预览，不创建）：{run_dir}')
            for step in steps:
                print(printable_command(step))
            return 0
        from .bootstrap import check_cuda, check_imports, environment_plan
        plan = environment_plan('glm_subj', experiment)
        python = configured_python(settings, 'glm_subj')
        print(f'  {plan.explanation}')
        ready, missing = check_imports(python, plan.imports)
        if not ready:
            raise ValueError('Glm-subj Python 环境缺少依赖: ' + ', '.join(missing))
        print(f'  [OK] 所需依赖已就绪：{python}')
        ready, detail = check_cuda(python, str(get_nested(settings, 'common.gpu', '0')))
        print(f"  [{'OK' if ready else 'FAIL'}] GPU：{detail}")
        if not ready:
            return 1
        doctor(settings, steps)
        task_summary(settings, method, experiment, mode, selection)
        if confirm is not None and not confirm():
            return 0
    except (OSError, ValueError, NotImplementedError) as exc:
        print(f'Glm-subj：{exc}')
        return 1
    print(f'\n运行目录：{run_dir}')
    print(f'日志文件：{run_dir / "run.log"}')
    try:
        run_dir.mkdir(parents=True)
    except OSError as exc:
        print(f'Glm-subj 输出目录不可写：{exc}')
        return 1
    manifest = {**identity(method, experiment), 'status': 'running',
                'run_directory': str(run_dir),
                'started_at': datetime.now(timezone.utc).isoformat(),
                'artifacts': artifacts, 'products': products,
                'commands': [step.command for step in steps]}
    manifest_file = run_dir / 'run_manifest.json'

    def save():
        temporary = manifest_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(manifest_file)

    save()
    try:
        import os
        with (run_dir / 'run.log').open('w', encoding='utf-8') as log:
            for step in steps:
                log.write(f"\n===== STEP: {step.name} =====\n")
                log.flush()
                env = {**os.environ, **step.env, 'PYTHONUNBUFFERED': '1'}
                process = subprocess.Popen(
                    step.command, cwd=step.cwd, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace', bufsize=1,
                )
                assert process.stdout is not None
                try:
                    for line in process.stdout:
                        safe_console_write(line)
                        log.write(line)
                        log.flush()
                    return_code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                finally:
                    process.stdout.close()
                if return_code:
                    raise subprocess.CalledProcessError(return_code, step.command)
        if not all(Path(p).is_file() for p in artifacts.values()):
            raise ValueError('运行未生成全部报告，不能标记为成功')
        manifest['status'] = 'success'
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError, KeyboardInterrupt) as exc:
        manifest.update(status='failed', error=str(exc))
        print(f'Glm-subj 失败；详情见 {run_dir / "run.log"}: {exc}')
        return 1
    finally:
        manifest['completed_at'] = datetime.now(timezone.utc).isoformat()
        save()
        print(f'运行记录：{manifest_file}')
