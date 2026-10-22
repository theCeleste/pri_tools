"""Isolated ChatGLM3/Subj command routing and manifest-backed history."""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import (PROJECT_ROOT, Step, configured_python, get_nested, method_env,
                   print_historical_results, printable_command, project_path, safe_console_write)

ROOT = PROJECT_ROOT / 'Glm-subj'
METHODS = {'cliptext', 'privsv', 'zo_dpga'}
DPSGD_ROOT = PROJECT_ROOT / 'dpsgd' / 'chatglm'


def runtime_profile(experiment):
    if experiment == 'indirect_attack':
        return 'glm_subj_indirect'
    return 'glm_subj_dpsgd' if experiment == 'performance_baseline' else 'glm_subj'


def config(settings, key, default=None):
    return get_nested(settings, f'glm_subj.{key}', default)


def path(settings, key, default):
    return project_path(config(settings, key, default)).resolve()


def identity(method, experiment):
    if method not in METHODS:
        raise ValueError('Glm-subj 仅支持 ClipText、PrivSV、ZO-DPGA')
    return dict(backend='glm_subj', method=method, model='chatglm3-6b', dataset='subj',
                experiment=experiment, role='paired' if experiment in {'direct_attack', 'indirect_attack'} else
                'baseline' if experiment in {'baseline', 'performance_baseline'} else 'protected')


def output_root(settings):
    return path(settings, 'output_root', 'Glm-subj/outputs')


def history(settings, method, experiment):
    """Select by manifest identity and recorded completion time, never directory mtime."""
    expected = identity(method, experiment)
    candidates = []
    for manifest in output_root(settings).glob('*/run_manifest.json'):
        try:
            data = json.loads(manifest.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                continue
            if data.get('status') != 'success' or any(data.get(k) != v for k, v in expected.items()):
                continue
            if experiment == 'indirect_attack':
                from .glm_subj_indirect import history_matches
                if not history_matches(data, settings, method):
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
    if experiment == 'indirect_attack':
        from .glm_subj_indirect import build_steps as build_indirect
        return build_indirect(settings, run_dir, method)
    if experiment == 'performance_baseline':
        return build_dpsgd_steps(settings, run_dir.resolve())
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


def build_dpsgd_steps(settings, run_dir):
    base = path(settings, 'base_model', get_nested(settings, 'common.chatglm3_6b', 'models/chatglm3-6b'))
    options = dict(batch_size=32, max_physical_batch_size=16, seq_length=128,
                   lora_rank=16, learning_rate=1e-4, max_grad_norm=3.0, epochs=4)
    values = config(settings, 'dpsgd', {})
    if not isinstance(values, dict) or set(values) - options.keys():
        raise ValueError('glm_subj.dpsgd 只能设置 batch_size、max_physical_batch_size、seq_length、'
                         'lora_rank、learning_rate、max_grad_norm、epochs')
    options.update(values)
    for key, value in options.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float('inf'):
            raise ValueError(f'glm_subj.dpsgd.{key} 必须为正数')
        if key not in {'learning_rate', 'max_grad_norm'} and not isinstance(value, int):
            raise ValueError(f'glm_subj.dpsgd.{key} 必须为正整数')
    gpu = str(get_nested(settings, 'common.gpu', '0'))
    if not gpu.strip() or ',' in gpu:
        raise ValueError('ChatGLM DP-SGD 仅支持单卡，请将 common.gpu 设为单个 GPU 编号')
    command = [configured_python(settings, runtime_profile('performance_baseline')),
               str(DPSGD_ROOT / 'dpsgd_chatglm3_subj.py'), '--model_name', str(base),
               '--output_dir', str(run_dir)]
    for key, value in options.items():
        command += ['--' + key, str(value)]
    return ([Step('ChatGLM Subj DP-SGD', command, DPSGD_ROOT,
                  method_env(settings, DPSGD_ROOT, PROJECT_ROOT))],
            {'training_report': str(run_dir / 'training_report.json')},
            {'checkpoints': str(run_dir)})


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
    if steps and str(PROJECT_ROOT / 'Glm-subj/Indirect_attack/run.py') in steps[0].command:
        from .glm_subj_indirect import doctor as indirect_doctor, METHODS as indirect_methods
        native = steps[0].command[steps[0].command.index('--method') + 1]
        method = next(m for m, values in indirect_methods.items() if values[0] == native)
        indirect_doctor(settings, method)
        return
    if steps and '--model_name' in steps[0].command:
        args = steps[0].command
        base = Path(args[args.index('--model_name') + 1])
        model_config = json.loads((base / 'config.json').read_text(encoding='utf-8'))
        if model_config.get('model_type') != 'chatglm' or not model_config.get('auto_map', {}).get('AutoModelForSequenceClassification'):
            raise ValueError('ChatGLM DP-SGD 需要含 AutoModelForSequenceClassification 映射的本地 ChatGLM 模型')
        return
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
    # if experiment == 'indirect_attack':
    #     from .glm_subj_indirect import resources
    #     info = resources(settings, method)
    #     # print('资源来源：' + ('随包权重复现' if info['resource_source'] == 'bundled' else '外部训练产物'))
    #     # print('保护/基线变体：' + str(info['variants']))
    #     if method == 'privsv':
    #         print('模型组成：ChatGLM 底座 + 运行时 Steering（不加载 LoRA）')
    mode_title = {'run': '立即执行', 'history': '查看最近历史结果', 'dry': '预览命令（不执行）'}[mode]
    print(f'模式：{mode_title}')


def show_history(settings, method, experiment, *, real_execution=False):
    history_root = project_path(str(settings.get('historical_results_dir', 'privacy_tools/historical_results')))
    title = {'cliptext': 'ClipText', 'privsv': 'PrivSV', 'zo_dpga': 'ZO-DPGA'}[method]
    print_historical_results(history_root / 'chatglm', method, title, experiment,
                             real_execution=real_execution)


def run(settings, method, experiment, mode='run', *, confirm=None, selection=None):
    identity(method, experiment)
    profile = runtime_profile(experiment)
    if experiment == 'indirect_attack':
        from .glm_subj_indirect import python as indirect_python
        python = indirect_python(settings)
    else:
        python = configured_python(settings, profile)
    if mode == 'history':
        print(f"  查看最近历史结果使用默认配置的 Python环境：{python}")
        print('  查看最近历史结果不会安装依赖或检查 GPU。')
        task_summary(settings, method, experiment, mode, selection)
        show_history(settings, method, experiment)
        return 0
    if experiment == 'indirect_attack':
        from .glm_subj_indirect import resources
        try:
            resources(settings, method)
        except (ValueError, TypeError, KeyError) as exc:
            print(f'Glm-subj：{exc}')
            return 1
    run_dir = output_root(settings) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    try:
        steps, artifacts, products = build_steps(settings, run_dir, method, experiment)
        if mode == 'dry':
            print(f"  命令预览使用默认配置的 Python环境：{python}")
            print('  命令预览不会检查 GPU 或执行实验。')
            task_summary(settings, method, experiment, mode, selection)
            print(f'\n运行目录（预览，不创建）：{run_dir}')
            for step in steps:
                print(printable_command(step).replace('dpzero', 'zo_dpga'))
            return 0
        from .bootstrap import check_cuda, check_imports, environment_plan
        plan = environment_plan(profile, experiment)
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
        extra_manifest = {}
        if experiment == 'indirect_attack':
            from .glm_subj_indirect import metadata
            extra_manifest = metadata(settings, method, fingerprints=True)
        task_summary(settings, method, experiment, mode, selection)
        if confirm is not None and not confirm():
            return 0
    except (OSError, ValueError, TypeError, KeyError, NotImplementedError) as exc:
        print(f'Glm-subj：{exc}')
        return 1
    print(f'\n运行目录：{run_dir}')
    print(f'日志文件：{run_dir / "run.log"}')
    try:
        run_dir.mkdir(parents=True)
    except OSError as exc:
        print(f'Glm-subj 输出目录不可写：{exc}')
        return 1
    manifest = {**identity(method, experiment), **extra_manifest, 'status': 'running',
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
                        safe_console_write(line.replace('dpzero', 'zo_dpga'))
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
        if experiment == 'indirect_attack':
            from .glm_subj_indirect import finalize
            result = finalize(settings, method, run_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        if not all(Path(p).is_file() for p in artifacts.values()):
            raise ValueError('运行未生成全部报告，不能标记为成功')
        manifest['status'] = 'success'
        show_history(settings, method, experiment, real_execution=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError, KeyboardInterrupt) as exc:
        manifest.update(status='failed', error=str(exc))
        print(f'Glm-subj 失败；详情见 {run_dir / "run.log"}: {exc}')
        return 1
    finally:
        manifest['completed_at'] = datetime.now(timezone.utc).isoformat()
        save()
        print(f'运行记录：{manifest_file}')
