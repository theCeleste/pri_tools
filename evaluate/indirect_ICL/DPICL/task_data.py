"""WikiText-2 input adaptation and reproducible local data snapshots."""
import argparse
import json

from datasets import Dataset, DatasetDict, load_dataset

from runtime_config import project_path

WIKITEXT_CONFIGS = ('wikitext-2-raw-v1', 'wikitext-2-v1')


def text_key(text):
    return ' '.join(text.split())


def prepare_wikitext2(data):
    seen = set()
    splits, counts = {}, {}
    # Earlier splits have priority, so no held-out text occurs in the private pool.
    for split in ('train', 'validation', 'test'):
        rows = []
        short = duplicates = 0
        for row in data[split]:
            text = row.get('text', row.get('sentence'))
            if not isinstance(text, str):
                raise ValueError(f'{split}: expected a string text field')
            text = text.strip()
            key = text_key(text)
            if not key or len(text) <= 10:
                short += 1
                continue
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            rows.append(text)
        splits[split] = Dataset.from_dict({'sentence': rows, 'label': [0] * len(rows)})
        counts[split] = dict(input=len(data[split]), kept=len(rows),
                             short_or_blank=short, duplicates=duplicates)
    return DatasetDict(splits), counts


def load_wikitext2(directory=None, config='wikitext-2-raw-v1'):
    if config not in WIKITEXT_CONFIGS:
        raise ValueError(f'Unsupported WikiText-2 config: {config}')
    if directory is None:
        data = load_dataset('wikitext', config)
    else:
        path = project_path(directory)
        metadata = path / 'dataset_config.json'
        if metadata.exists():
            saved = json.loads(metadata.read_text(encoding='utf-8'))
            if saved['config'] != config:
                raise ValueError('Local WikiText-2 snapshot config does not match --wikitext2-config')
        data = DatasetDict()
        for split in ('train', 'validation', 'test'):
            entries = json.loads((path / f'{split}.json').read_text(encoding='utf-8'))
            texts = [row if isinstance(row, str) else row.get('text', row.get('sentence'))
                     for row in entries]
            data[split] = Dataset.from_dict({'text': texts})
    return prepare_wikitext2(data)


def generation_full_datum(labels, datum):
    if labels == ['text']:
        return f'Text: "{datum["sentence"]} END"\n'
    return f'Sentiment: "{labels[datum["label"]]}"\nText: "{datum["sentence"] + " END"}"\n'


def generation_test_input(labels, label):
    if labels == ['text']:
        return 'Text: "'
    return f'Sentiment: "{labels[label]}"\nText: "'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wikitext2-dir')
    parser.add_argument('--wikitext2-config', choices=WIKITEXT_CONFIGS, default=WIKITEXT_CONFIGS[0])
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    data, counts = load_wikitext2(args.wikitext2_dir, args.wikitext2_config)
    destination = project_path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    for split in data:
        (destination / f'{split}.json').write_text(
            json.dumps(list(data[split]['sentence']), ensure_ascii=False, indent=2), encoding='utf-8')
    (destination / 'dataset_config.json').write_text(json.dumps(dict(
        dataset='wikitext2', config=args.wikitext2_config,
        source=args.wikitext2_dir or 'wikitext', counts=counts,
        filter='len(text.strip()) > 10', deduplication='whitespace-normalized, train/validation/test priority',
        label='internal constant 0; no semantic class labels'), indent=2), encoding='utf-8')
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
