"""SUBJ indirect evaluation adapter. Importing this module needs only the standard library."""
from __future__ import annotations

import bisect
import csv
import re
import hashlib
import json
import math
import random
from pathlib import Path

from .core import PROJECT_ROOT, Step, configured_python, get_nested, method_env, project_path

ROOT = PROJECT_ROOT / 'Glm-subj' / 'Indirect_attack'
BACKEND = 'glm_subj_indirect'
SCHEMA = 1
METHODS = {'cliptext': ('embedding', 'raw', 'cliptext_lora'),
           'privsv': ('example', 'mean', 'private'),
           'zo_dpga': ('gradient', 'mezo', 'dpzero')}
TARGETS = (1., .5, .1, .01, .001, .0001)


def read_json(file):
    return json.loads(Path(file).read_text(encoding='utf-8'))


def digest(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def python(settings):
    return str(get_nested(settings, 'python.glm_subj_indirect') or configured_python(settings, 'glm_subj'))


def resources(settings, method):
    native, baseline, protected = METHODS[method]
    cfg = get_nested(settings, 'glm_subj.indirect', {})
    if not isinstance(cfg, dict) or set(cfg) - {'resource_source', 'proposal_model', 'neighbors', 'embedding', 'gradient', 'example'}:
        raise ValueError('glm_subj.indirect 配置无效或包含未知参数')
    source = cfg.get('resource_source', 'bundled')
    if source not in {'bundled', 'external'}:
        raise ValueError('glm_subj.indirect.resource_source 必须为 bundled 或 external')
    local = cfg.get(native, {})
    allowed = {'interface_dir'} if native == 'example' else {'baseline_adapter', 'protected_adapter', 'splits', 'provenance'}
    if not isinstance(local, dict) or set(local) - allowed:
        raise ValueError(f'glm_subj.indirect.{native} 配置无效或包含未知参数')
    if source == 'bundled' and local:
        raise ValueError('覆盖随包资源前必须显式设置 resource_source=external')
    def path(value):
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError('资源路径必须为非空字符串')
        return str(project_path(value).resolve())
    value = dict(evaluation_backend=BACKEND, schema_version=SCHEMA, resource_source=source,
                 native_method=native, variants={'baseline': baseline, 'protected': protected},
                 base_model=path(get_nested(settings, 'glm_subj.base_model',
                                  get_nested(settings, 'common.chatglm3_6b', 'models/chatglm3-6b'))),
                 proposal_model=path(cfg.get('proposal_model', ROOT / 'resources/bert-base-uncased')))
    if native == 'example':
        if source == 'external' and not local.get('interface_dir'):
            raise ValueError('external example 需要 interface_dir（含配套接口代码与构造/测试数据）')
        value['interface_dir'] = path(local.get('interface_dir', ROOT / 'example/interface'))
    else:
        for key, default in [('baseline_adapter', ROOT / native / 'weights' / baseline),
                             ('protected_adapter', ROOT / native / 'weights' / protected),
                             ('splits', ROOT / native / 'data/splits')]:
            if source == 'external' and not local.get(key):
                raise ValueError(f'external {native} 缺少 {key}')
            value[key] = path(local.get(key, default))
        if source == 'external':
            if not local.get('provenance'):
                raise ValueError('外部权重需要 provenance 训练来源记录，不能沿用随包成员身份')
            value['provenance'] = path(local['provenance'])
    if cfg.get('neighbors'):
        value['neighbors'] = path(cfg['neighbors'])
    return value


def build_steps(settings, run_dir, method):
    info = resources(settings, method)
    gpu = str(get_nested(settings, 'common.gpu', '0'))
    if not gpu.isascii() or not gpu.isdecimal():
        raise ValueError('common.gpu 必须为单个非负物理 GPU 编号')
    mode = 'attack' if info.get('neighbors') and info['native_method'] != 'example' else 'full'
    command = [python(settings), str(ROOT / 'run.py'), '--mode', mode,
               '--method', info['native_method'], '--variant', 'both', '--gpu', gpu,
               '--model', info['base_model'], '--output', str(Path(run_dir).resolve() / 'indirect'),
               '--proposal-model', info['proposal_model']]
    for key in ('baseline_adapter', 'protected_adapter', 'splits', 'interface_dir', 'neighbors'):
        if key in info:
            command += ['--' + key.replace('_', '-'), info[key]]
    env = method_env(settings, PROJECT_ROOT)
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    artifacts = {role + '_metrics': str(Path(run_dir).resolve() / 'indirect' / variant / 'evaluation/metrics.json')
                 for role, variant in info['variants'].items()}
    artifacts['indirect_summary'] = str(Path(run_dir).resolve() / 'indirect_summary.json')
    return [Step('Subj 间接评估（保护/基线成对）', command, ROOT, env)], artifacts, {}


def rows(file):
    return [json.loads(line) for line in Path(file).read_text(encoding='utf-8').splitlines() if line.strip()]


def normalize(text):
    return ' '.join(str(text).lower().split())


def validate_groups(groups):
    seen = set()
    for name, records in groups.items():
        texts = [normalize(r['text']) for r in records]
        if (not texts and name != 'dev') or any(not t for t in texts) or len(texts) != len(set(texts)):
            raise ValueError(f'{name} 含空文本、空划分或重复正文')
        if seen.intersection(texts):
            raise ValueError(f'{name} 与其他划分存在交集')
        seen.update(texts)


def samples(info):
    if info['native_method'] != 'example':
        splits = Path(info['splits'])
        groups = {k: rows(splits / f'{v}.jsonl') for k, v in
                  [('member', 'train'), ('dev', 'dev'), ('nonmember', 'eval')]}
        validate_groups(groups)
        if (len(groups['member']), len(groups['nonmember'])) != (100, 1000):
            raise ValueError('正式 embedding/gradient 需要 100 member + 1000 nonmember')
        for name, records in groups.items():
            if any(r.get('membership') != name for r in records):
                raise ValueError(f'{name} membership 标签错误')
            keys = [r['source_index'] for r in records]
            if len(keys) != len(set(keys)):
                raise ValueError(f'{name} source_index 重复')
        return {k: groups[k] for k in ('member', 'nonmember')}
    directory = Path(info['interface_dir']) / 'data'
    construction = read_json(directory / 'generate_dataset.json')
    test = read_json(directory / 'test_dataset_ab.json')
    if len(construction) < 1000 or len(test) < 2000:
        raise ValueError('example 需要至少 1000 构造记录和 2000 nonmember 候选')
    # Mirror native load_samples()/parse_record(); use the exact same selection order.
    selected = list(enumerate(construction))
    random.Random(42).shuffle(selected)
    def convert(record, index):
        question = record['question']
        match = re.search(r'Sentence: "(.*)"\n\nOptions:', question, re.DOTALL)
        if not match:
            raise ValueError('example question 缺少 SUBJ Sentence/Options 正文')
        for field in ('answer_matching_behavior', 'answer_not_matching_behavior'):
            if not isinstance(record.get(field), str) or not record[field].startswith(('(A)', '(B)')):
                raise ValueError('example answer 字段无效')
        return {'text': match.group(1), 'source_index': index}
    groups = {'member': [convert(r, i) for i, r in selected[:1000]],
              'nonmember': [convert(r, i) for i, r in enumerate(test[:2000])]}
    validate_groups(groups)
    return groups


def required_files(info):
    files = [Path(info['base_model']) / n for n in ('config.json', 'tokenizer_config.json', 'tokenizer.model')]
    base = Path(info['base_model'])
    index = base / 'model.safetensors.index.json'
    if index.is_file():
        files += [index] + [base / p for p in sorted(set(read_json(index)['weight_map'].values()))]
    else:
        files += [base / 'model.safetensors']
    proposal = Path(info['proposal_model'])
    files += [proposal / n for n in ('config.json', 'tokenizer_config.json', 'vocab.txt', 'model.safetensors')]
    config = base / 'config.json'
    if config.is_file():
        for reference in read_json(config).get('auto_map', {}).values():
            for name in reference if isinstance(reference, list) else [reference]:
                if isinstance(name, str):
                    files.append(base / (name.rsplit('.', 1)[0] + '.py'))
    if info['native_method'] == 'example':
        interface = Path(info['interface_dir'])
        files += [interface / n for n in ('data/generate_dataset.json', 'data/test_dataset_ab.json',
                                          'subj_steered_model/__init__.py', 'subj_steered_model/interface.py',
                                          'subj_steered_model/steering.py')]
    else:
        for key in ('baseline_adapter', 'protected_adapter'):
            files += [Path(info[key]) / n for n in ('adapter_config.json', 'adapter_model.safetensors')]
        files += [Path(info['splits']) / f'{name}.jsonl' for name in ('train', 'dev', 'eval')]
    for key in ('neighbors', 'provenance'):
        if key in info:
            files.append(Path(info[key]))
    if info.get('neighbors'):
        files.append(Path(info['neighbors'] + '.config.json'))
    return files


def validate_provenance(info, groups):
    """Evidence binds each actual adapter to original training texts, including ClipText mappings."""
    if 'provenance' not in info:
        return
    proof = read_json(info['provenance'])
    for role in ('baseline', 'protected'):
        record = proof.get(role, {})
        adapter_hash = digest(Path(info[role + '_adapter']) / 'adapter_model.safetensors')
        if record.get('adapter_sha256') != adapter_hash:
            raise ValueError(f'provenance {role} adapter 指纹不匹配')
        training = record.get('original_training_texts')
        if not isinstance(training, list) or not training or any(not isinstance(t, str) or not t.strip() for t in training):
            raise ValueError(f'provenance {role} 缺少 original_training_texts')
        known = set(map(normalize, training))
        if any(normalize(r['text']) not in known for r in groups['member']):
            raise ValueError(f'{role} member 不属于记录的训练数据')
        if any(normalize(r['text']) in known for r in groups['nonmember']):
            raise ValueError(f'{role} nonmember 出现在训练数据中')


def validate_neighbors(info, groups):
    if not info.get('neighbors'):
        return
    cache = rows(info['neighbors'])
    key_field = 'source_index' if info['native_method'] == 'example' else 'sample_index'
    lookup = {(r['membership'], int(r[key_field])): r for r in cache}
    expected = {(membership, int(row['source_index']) if key_field == 'source_index' else i): row
                for membership, group in groups.items() for i, row in enumerate(group)}
    if len(lookup) != len(cache) or set(lookup) != set(expected):
        raise ValueError('邻居缓存的样本 ID 与攻击划分不一致')
    for key, row in expected.items():
        item = lookup[key]
        if item.get('text') != row['text'] or not isinstance(item.get('neighbors'), list) or len(item['neighbors']) != 10:
            raise ValueError(f'邻居缓存正文或邻居数量不匹配: {key}')
        if any(not isinstance(t, str) or not t.strip() for t in item['neighbors']):
            raise ValueError('邻居缓存含空文本')
    config = read_json(info['neighbors'] + '.config.json')
    expected_config = dict(seed=43 if key_field == 'source_index' else 42, proposal_max_length=512,
                           neighbor_top_k=6, candidate_neighbors=50, num_neighbors=10)
    if key_field == 'sample_index':
        expected_config['dropout'] = .7
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise ValueError(f'邻居缓存 {key} 不匹配')
    if str(Path(config.get('proposal_model', '')).resolve()) != info['proposal_model']:
        raise ValueError('邻居缓存 proposal_model 路径不匹配')


def doctor(settings, method):
    info = resources(settings, method)
    for file in required_files(info):
        if not file.is_file() or (not file.stat().st_size and file.name != 'dev.jsonl'):
            raise ValueError(f'间接评估资源缺失或为空: {file}')
    if read_json(Path(info['base_model']) / 'config.json').get('model_type') != 'chatglm':
        raise ValueError('间接评估底座必须为 ChatGLM')
    if read_json(Path(info['proposal_model']) / 'config.json').get('model_type') != 'bert':
        raise ValueError('间接评估 proposal 必须为 BERT')
    if info['native_method'] != 'example':
        for role in ('baseline', 'protected'):
            adapter = read_json(Path(info[role + '_adapter']) / 'adapter_config.json')
            if adapter.get('task_type') != 'CAUSAL_LM' or adapter.get('peft_type') != 'LORA':
                raise ValueError(f'{role} 必须为 causal LoRA adapter')
    groups = samples(info)
    validate_provenance(info, groups)
    validate_neighbors(info, groups)
    return info


def metadata(settings, method, *, fingerprints=False):
    info = resources(settings, method)
    if fingerprints:
        info['gpu'] = str(get_nested(settings, 'common.gpu', '0'))
        info['python'] = python(settings)
        info['sha256'] = {str(file): digest(file) for file in required_files(info)}
    return info


def history_matches(data, settings, method):
    try:
        expected = metadata(settings, method)
        return all(data.get(k) == v for k, v in expected.items())
    except (ValueError, TypeError, KeyError):
        return False


def finalize(settings, method, run_dir):
    """Validate complete paired scores and native metrics before the runner marks success."""
    info = resources(settings, method)
    output = Path(run_dir) / 'indirect'
    counts = (1000, 1000) if info['native_method'] == 'example' else (100, 1000)
    results, identities, metric_warnings = {}, [], []
    expected_groups = samples(info)
    expected_text = {(membership, str(r['source_index'])): r['text']
                     for membership, group in expected_groups.items() for r in group}
    for role, variant in info['variants'].items():
        folder = output / variant
        with (folder / 'scores.csv').open(newline='', encoding='utf-8') as handle:
            scores = list(csv.DictReader(handle))
        grouped = {k: [r for r in scores if r['membership'] == k] for k in ('member', 'nonmember')}
        if len(scores) != sum(counts) or tuple(map(len, grouped.values())) != counts:
            raise ValueError('间接评估分数不完整或标签无效')
        identity = sorted((r['membership'], int(r['sample_index']), int(r['source_index']), r['text'],
                           int(r['label']), int(r['num_neighbors'])) for r in scores)
        if len({(r[0], r[1]) for r in identity}) != len(identity) or any(r[-1] != 10 for r in identity):
            raise ValueError('攻击样本重复或邻居数量错误')
        identities.append(identity)
        for membership, group in grouped.items():
            if sorted(int(r['sample_index']) for r in group) != list(range(len(group))):
                raise ValueError('攻击 sample_index 不连续')
            if len({r['source_index'] for r in group}) != len(group):
                raise ValueError('攻击 source_index 重复')
        for row in scores:
            if expected_text.get((row['membership'], row['source_index'])) != row['text']:
                raise ValueError('分数中的正文或来源 ID 与实际攻击划分不一致')
            if not all(math.isfinite(float(row[k])) for k in ('score', 'original_score', 'neighbor_mean')):
                raise ValueError('间接评估分数含 NaN/Inf')
            if not math.isclose(float(row['score']), float(row['original_score']) - float(row['neighbor_mean']), abs_tol=1e-7):
                raise ValueError('攻击分数不等于原文与邻居的 logprob 差')
        if info['native_method'] != 'example':
            config = read_json(folder / 'run_config.json')
            expected_neighbors = info.get('neighbors', str(output.resolve() / 'neighbors.jsonl'))
            if config.get('neighbors') != expected_neighbors or config.get('adapter') != info[role + '_adapter']:
                raise ValueError('保护/基线未使用同一邻居文件或配置的 adapter')
        failures = folder / 'failures.json'
        if failures.exists() and read_json(failures):
            raise ValueError(f'{variant} 存在攻击失败记录')
        result = read_json(folder / 'evaluation/metrics.json')
        if (result.get('member_count'), result.get('nonmember_count')) != counts:
            raise ValueError('原生 metrics 样本数与实际分数不匹配')
        # Rank AUC (ties receive half credit), independent of numpy/sklearn in the UI environment.
        member = [float(r['score']) for r in grouped['member']]
        nonmember = [float(r['score']) for r in grouped['nonmember']]
        ordered = sorted(nonmember)
        auc = sum((bisect.bisect_left(ordered, x) + bisect.bisect_right(ordered, x)) / 2
                  for x in member) / (len(member) * len(nonmember))
        if not math.isclose(result['auc'], auc, abs_tol=1e-10):
            raise ValueError('原生 AUC 与 scores 不一致')
        metrics = result.get('metrics', [])
        if len(metrics) != len(TARGETS):
            raise ValueError('原生 metrics 缺少目标 FPR')
        for row, target in zip(metrics, TARGETS):
            threshold = row['threshold']
            if not math.isfinite(threshold):
                raise ValueError('阈值非有限数值')
            tp, fp = sum(x > threshold for x in member), sum(x > threshold for x in nonmember)
            expected = dict(target_fpr=target, actual_fpr=fp / len(nonmember), tpr=tp / len(member),
                            tp=tp, fp=fp, tn=len(nonmember)-fp, fn=len(member)-tp, auc=auc,
                            member_count=len(member), nonmember_count=len(nonmember))
            mismatches = [k for k, value in expected.items()
                          if k not in row or not math.isclose(row[k], value, abs_tol=1e-10)]
            if 'actual_fpr' in row and row['actual_fpr'] > target:
                mismatches.append('actual_fpr_exceeds_target')
            if mismatches:
                # Native metric code and this validator may parse CSV floats with
                # different precision. Preserve the native result, but surface
                # discrepancies without failing an otherwise complete run.
                metric_warnings.append({
                    'role': role,
                    'target_fpr': target,
                    'mismatches': mismatches,
                })
        results[role] = result
    if identities[0] != identities[1]:
        raise ValueError('保护与基线攻击样本不匹配')
    if info['native_method'] == 'example':
        selection = read_json(output / 'selection.json')
        if selection.get('subset_seed') != 60:
            raise ValueError('example 子集 seed 错误')
        for membership in ('member', 'nonmember'):
            actual = sorted(r[2] for r in identities[0] if r[0] == membership)
            if sorted(selection.get(membership + '_source_indices', [])) != actual:
                raise ValueError('example 子集记录与实际 scores 不一致')
        if (output / 'full_pool/failures.json').exists() and read_json(output / 'full_pool/failures.json'):
            raise ValueError('example 完整攻击池存在失败记录')
    summary = {**info, 'results': results,
               'protected_minus_baseline_auc': results['protected']['auc'] - results['baseline']['auc'],
               'fpr_resolution': 1 / counts[1],
               'metric_validation_warnings': metric_warnings,
               'model_composition': 'base + runtime steering (no LoRA)' if method == 'privsv' else 'base + LoRA'}
    file = Path(run_dir) / 'indirect_summary.json'
    file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary
