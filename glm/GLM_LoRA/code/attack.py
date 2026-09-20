import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

try:
    from peft import AutoPeftModelForCausalLM, PeftModel
except ImportError:  # pragma: no cover - kept for full-model-only environments.
    AutoPeftModelForCausalLM = None
    PeftModel = None


SST2_LABEL_TEXT = {
    "0": "negative",
    "1": "positive",
    0: "negative",
    1: "positive",
    "negative": "negative",
    "positive": "positive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate neighbour-based MIA scores for a causal LM. "
            "Supports full models and PEFT/LoRA adapters."
        )
    )

    parser.add_argument(
        "--target-model-path",
        required=True,
        help="Base/full causal LM path or repo id. If it is a PEFT adapter, AutoPeftModel is used.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional PEFT/LoRA adapter path. Use with --target-model-path as the base model.",
    )
    parser.add_argument(
        "--target-tokenizer-path",
        default=None,
        help="Tokenizer path. Defaults to adapter path when set, otherwise target model path.",
    )
    parser.add_argument(
        "--neighbor-model-path",
        required=True,
        help="Masked-LM path/repo used to generate neighbours, for example roberta-base.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to target tokenizer/model loading.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load already downloaded Hugging Face files.",
    )
    parser.add_argument(
        "--target-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype for the target model.",
    )
    parser.add_argument(
        "--target-device-map",
        default="auto",
        help='Transformers device_map for the target model, or "none" to call model.to(device).',
    )
    parser.add_argument(
        "--target-device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for target inputs and for target model when --target-device-map none.",
    )
    parser.add_argument(
        "--neighbor-device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for the masked neighbour model.",
    )

    data = parser.add_argument_group("dataset")
    data.add_argument("--dataset-dir", default=None, help="HF load_from_disk directory.")
    data.add_argument("--dataset-file", default=None, help="CSV/TSV/JSONL file with text rows.")
    data.add_argument("--dataset-name", default=None, help="Optional Hugging Face dataset name.")
    data.add_argument("--dataset-config", default=None, help="Optional Hugging Face dataset config.")
    data.add_argument("--split", default="train", help="Split name to score.")
    data.add_argument("--text-column", default="sentence", help="Column containing source text.")
    data.add_argument("--label-column", default="label", help="Column containing labels.")
    data.add_argument("--start-index", type=int, default=0, help="Start index inside the split.")
    data.add_argument("--end-index", type=int, default=None, help="Exclusive end index inside the split.")
    data.add_argument("--num-samples", type=int, default=None, help="Number of samples to score.")
    data.add_argument("--proc-id", type=int, default=None, help="Shard id for multi-process runs.")
    data.add_argument(
        "--samples-per-shard",
        type=int,
        default=500,
        help="Samples per shard when --proc-id is set.",
    )

    text = parser.add_argument_group("target text formatting")
    text.add_argument(
        "--text-template",
        choices=(
            "sentence",
            "sst2-review-sentiment",
            "agnews-article-topic",
            "wikitext-text",
            "custom",
        ),
        default="sentence",
        help=(
            "How to turn dataset rows into target LM text. "
            "Use fixed templates to keep MIA scoring aligned with fine-tuning."
        ),
    )
    text.add_argument(
        "--custom-template",
        default=None,
        help="Python format string for --text-template custom. Keys: text, label, label_text, eos.",
    )
    text.add_argument(
        "--add-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append target tokenizer eos token when available.",
    )

    attack = parser.add_argument_group("attack")
    attack.add_argument("--max-length", type=int, default=256, help="Max token length.")
    attack.add_argument("--neighbor-top-k", type=int, default=8, help="Top replacements per token.")
    attack.add_argument("--max-neighbors", type=int, default=50, help="Max neighbours per sample.")
    attack.add_argument(
        "--max-neighbor-positions",
        type=int,
        default=None,
        help="Optional cap on token positions considered for replacement.",
    )
    attack.add_argument("--score-batch-size", type=int, default=8, help="Target scoring batch size.")
    attack.add_argument("--output-pkl", required=True, help="Where to save score pickle.")
    attack.add_argument("--metadata-json", default=None, help="Optional metadata JSON output path.")
    attack.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse an existing output pkl and continue from the next unsaved row.",
    )
    attack.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Save partial pkl every N newly scored rows when resuming/caching.",
    )

    return parser.parse_args()


def torch_dtype_from_name(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def adapter_config_exists(path: str) -> bool:
    return Path(path).joinpath("adapter_config.json").exists()


def tokenizer_default_path(args: argparse.Namespace) -> str:
    if args.target_tokenizer_path:
        return args.target_tokenizer_path
    if args.adapter_path:
        return args.adapter_path
    return args.target_model_path


def ensure_pad_token(tokenizer: Any, model: Optional[Any] = None) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    if model is not None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model is not None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id


def first_model_device(model: Any, fallback: str) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for device in hf_device_map.values():
            if device not in ("cpu", "disk"):
                return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    dtype = torch_dtype_from_name(args.target_dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if args.target_device_map != "none":
        kwargs["device_map"] = args.target_device_map
    return kwargs


def load_target(args: argparse.Namespace) -> Tuple[Any, Any, torch.device]:
    tokenizer_path = tokenizer_default_path(args)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    kwargs = model_kwargs(args)
    target_path = args.target_model_path

    if args.adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is required when --adapter-path is used.")
        base_model = AutoModelForCausalLM.from_pretrained(target_path, **kwargs)
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            is_trainable=False,
        )
    elif adapter_config_exists(target_path):
        if AutoPeftModelForCausalLM is None:
            raise RuntimeError("peft is required to load a PEFT adapter as the target model.")
        model = AutoPeftModelForCausalLM.from_pretrained(target_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(target_path, **kwargs)

    ensure_pad_token(tokenizer, model)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.target_device_map == "none":
        model = model.to(args.target_device)

    model.eval()
    device = first_model_device(model, args.target_device)
    return tokenizer, model, device


def load_neighbor(args: argparse.Namespace) -> Tuple[Any, Any, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.neighbor_model_path,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.neighbor_model_path,
        local_files_only=args.local_files_only,
    )
    if tokenizer.mask_token_id is None:
        raise ValueError(
            f"Neighbour model {args.neighbor_model_path!r} has no mask token; use a masked LM."
        )
    model = model.to(args.neighbor_device)
    model.eval()
    return tokenizer, model, torch.device(args.neighbor_device)


def read_dataset_file(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    import pandas as pd

    if suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t")
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".json":
        frame = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported dataset file suffix: {suffix}")
    return frame.to_dict(orient="records")


def compute_slice(total: int, args: argparse.Namespace) -> Tuple[int, int]:
    if args.proc_id is not None:
        start = args.proc_id * args.samples_per_shard
        end = start + args.samples_per_shard
    else:
        start = args.start_index
        if args.end_index is not None:
            end = args.end_index
        elif args.num_samples is not None:
            end = start + args.num_samples
        else:
            end = total

    start = max(0, min(start, total))
    end = max(start, min(end, total))
    return start, end


def load_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if args.dataset_file:
        all_rows = read_dataset_file(Path(args.dataset_file))
    elif args.dataset_dir:
        dataset_path = Path(args.dataset_dir)
        split_path = dataset_path / args.split
        if split_path.exists():
            dataset = load_from_disk(str(split_path))
        else:
            loaded = load_from_disk(str(dataset_path))
            if isinstance(loaded, DatasetDict):
                dataset = loaded[args.split]
            else:
                dataset = loaded
        all_rows = [dataset[i] for i in range(len(dataset))]
    elif args.dataset_name:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
        all_rows = [dataset[i] for i in range(len(dataset))]
    else:
        raise ValueError("Provide one of --dataset-dir, --dataset-file, or --dataset-name.")

    start, end = compute_slice(len(all_rows), args)
    return all_rows[start:end], {
        "total_rows": len(all_rows),
        "start_index": start,
        "end_index": end,
        "selected_rows": end - start,
    }


def label_text(label: Any) -> str:
    if label is None:
        return ""
    key = label
    if isinstance(label, str):
        key = label.strip().lower()
    return SST2_LABEL_TEXT.get(key, str(label))


def format_target_text(
    source_text: str,
    row: Dict[str, Any],
    args: argparse.Namespace,
    tokenizer: Any,
) -> str:
    label = row.get(args.label_column, "")
    label_as_text = label_text(label)
    eos = tokenizer.eos_token or ""

    if args.text_template == "sentence":
        rendered = source_text.strip()
    elif args.text_template == "sst2-review-sentiment":
        rendered = f"Review: {source_text.strip()}\nSentiment: {label_as_text}"
    elif args.text_template == "agnews-article-topic":
        rendered = f"Article: {source_text.strip()}\nTopic: {label_as_text}"
    elif args.text_template == "wikitext-text":
        rendered = source_text.strip()
    elif args.text_template == "custom":
        if not args.custom_template:
            raise ValueError("--custom-template is required when --text-template custom is used.")
        rendered = args.custom_template.format(
            text=source_text.strip(),
            label=label,
            label_text=label_as_text,
            eos=eos,
        )
    else:
        raise ValueError(f"Unknown text template: {args.text_template}")

    if args.add_eos and eos and not rendered.endswith(eos):
        rendered += eos
    return rendered


def chunks(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


@torch.no_grad()
def score_texts(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int,
) -> List[float]:
    if not texts:
        return []

    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    labels = input_ids.clone()
    if attention_mask is not None:
        labels[attention_mask == 0] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    vocab_size = logits.shape[-1]

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    token_loss = loss_fct(
        logits.reshape(-1, vocab_size),
        shifted_labels.reshape(-1),
    ).reshape(shifted_labels.shape)

    valid = shifted_labels.ne(-100)
    denom = valid.sum(dim=1).clamp_min(1)
    mean_loss = (token_loss * valid).sum(dim=1) / denom
    return (-mean_loss).detach().cpu().tolist()


@torch.no_grad()
def generate_neighbours(
    source_text: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> List[Tuple[str, float]]:
    encoded = tokenizer(
        source_text,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    special_mask = encoded.get("special_tokens_mask")
    if special_mask is None:
        special_mask = torch.zeros_like(input_ids)

    positions: List[int] = []
    for index, is_special in enumerate(special_mask[0].tolist()):
        token_id = int(input_ids[0, index])
        if is_special:
            continue
        if token_id in tokenizer.all_special_ids:
            continue
        positions.append(index)

    if args.max_neighbor_positions is not None:
        positions = positions[: args.max_neighbor_positions]

    candidates: Dict[str, float] = {}
    mask_id = tokenizer.mask_token_id

    for position in positions:
        original_id = int(input_ids[0, position])
        masked_ids = input_ids.clone()
        masked_ids[0, position] = mask_id

        outputs = model(input_ids=masked_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, position].float()
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=args.neighbor_top_k)

        for prob, candidate_id in zip(top_probs.tolist(), top_ids.tolist()):
            candidate_id = int(candidate_id)
            if candidate_id == original_id:
                continue
            if candidate_id in tokenizer.all_special_ids:
                continue
            if candidate_id == mask_id:
                continue

            alt_ids = input_ids.clone()
            alt_ids[0, position] = candidate_id
            alt_text = tokenizer.decode(
                alt_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if not alt_text or alt_text == source_text.strip():
                continue
            candidates[alt_text] = max(candidates.get(alt_text, 0.0), float(prob))

    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    return ranked[: args.max_neighbors]


def score_one_row(
    row: Dict[str, Any],
    args: argparse.Namespace,
    target_tokenizer: Any,
    target_model: Any,
    target_device: torch.device,
    neighbor_tokenizer: Any,
    neighbor_model: Any,
    neighbor_device: torch.device,
) -> Optional[Dict[Any, Any]]:
    raw_text = row.get(args.text_column)
    if raw_text is None:
        return None
    source_text = str(raw_text).replace("\\", " ").strip()
    if not source_text:
        return None

    original_target_text = format_target_text(source_text, row, args, target_tokenizer)
    original_score = score_texts(
        [original_target_text],
        target_tokenizer,
        target_model,
        target_device,
        args.max_length,
    )[0]

    scores: Dict[Any, Any] = {f"<original_text>: {original_target_text}": original_score}
    neighbours = generate_neighbours(
        source_text,
        neighbor_tokenizer,
        neighbor_model,
        neighbor_device,
        args,
    )

    neighbour_target_texts = [
        format_target_text(neighbour_text, row, args, target_tokenizer)
        for neighbour_text, _ in neighbours
    ]
    neighbour_scores: List[float] = []
    for batch in chunks(neighbour_target_texts, args.score_batch_size):
        neighbour_scores.extend(
            score_texts(
                batch,
                target_tokenizer,
                target_model,
                target_device,
                args.max_length,
            )
        )

    for (neighbour_text, replacement_score), target_text, score in zip(
        neighbours,
        neighbour_target_texts,
        neighbour_scores,
    ):
        scores[(target_text, replacement_score)] = score

    return scores


def write_metadata(path: str, args: argparse.Namespace, slice_meta: Dict[str, Any], count: int) -> None:
    metadata = {
        "target_model_path": args.target_model_path,
        "adapter_path": args.adapter_path,
        "target_tokenizer_path": tokenizer_default_path(args),
        "neighbor_model_path": args.neighbor_model_path,
        "dataset_dir": args.dataset_dir,
        "dataset_file": args.dataset_file,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "text_column": args.text_column,
        "label_column": args.label_column,
        "text_template": args.text_template,
        "add_eos": args.add_eos,
        "max_length": args.max_length,
        "neighbor_top_k": args.neighbor_top_k,
        "max_neighbors": args.max_neighbors,
        "score_batch_size": args.score_batch_size,
        "slice": slice_meta,
        "valid_score_rows": count,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def load_cached_scores(path: Path) -> List[Dict[Any, Any]]:
    with path.open("rb") as handle:
        cached = pickle.load(handle)
    if not isinstance(cached, list):
        raise ValueError(f"Cache file is not a list: {path}")
    return cached


def save_scores(path: Path, scores: List[Dict[Any, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(scores, handle)
    temp_path.replace(path)


def main() -> int:
    args = parse_args()

    print("=" * 80)
    print("Generic neighbour MIA score generation")
    print("=" * 80)
    print(f"Target model    : {args.target_model_path}")
    print(f"Adapter         : {args.adapter_path or '<none>'}")
    print(f"Target tokenizer: {tokenizer_default_path(args)}")
    print(f"Neighbour model : {args.neighbor_model_path}")
    print(f"Dataset split   : {args.split}")
    print(f"Text template   : {args.text_template}")

    rows, slice_meta = load_rows(args)
    print(f"Rows selected   : {slice_meta['selected_rows']} / {slice_meta['total_rows']}")
    if not rows:
        raise RuntimeError("No rows selected for scoring.")

    output_path = Path(args.output_pkl)
    all_scores: List[Dict[Any, Any]] = []
    resume_offset = 0

    if args.resume and output_path.exists():
        all_scores = load_cached_scores(output_path)
        resume_offset = len(all_scores)
        print(f"Cache pkl        : {output_path}")
        print(f"Cached rows      : {resume_offset}")

        if resume_offset >= len(rows):
            print("Cache is complete; skipping model loading and scoring.")
            if args.metadata_json:
                write_metadata(args.metadata_json, args, slice_meta, len(all_scores))
                print(f"Saved metadata   : {args.metadata_json}")
            return 0

        print(f"Resuming at row   : {resume_offset}")

    target_tokenizer, target_model, target_device = load_target(args)
    neighbor_tokenizer, neighbor_model, neighbor_device = load_neighbor(args)

    skipped = 0
    scored_since_checkpoint = 0
    for row in tqdm(rows[resume_offset:], desc=f"scoring {args.split}"):
        result = score_one_row(
            row,
            args,
            target_tokenizer,
            target_model,
            target_device,
            neighbor_tokenizer,
            neighbor_model,
            neighbor_device,
        )
        if result is None:
            skipped += 1
            continue
        all_scores.append(result)
        scored_since_checkpoint += 1

        if args.checkpoint_every > 0 and scored_since_checkpoint >= args.checkpoint_every:
            save_scores(output_path, all_scores)
            scored_since_checkpoint = 0

    save_scores(output_path, all_scores)

    print("=" * 80)
    print(f"Saved pkl        : {output_path}")
    print(f"Valid rows       : {len(all_scores)}")
    print(f"Skipped rows     : {skipped}")
    if all_scores:
        neighbour_counts = [max(0, len(scores) - 1) for scores in all_scores]
        print(f"Mean neighbours  : {sum(neighbour_counts) / len(neighbour_counts):.2f}")
        print(f"Min/Max neighbours: {min(neighbour_counts)} / {max(neighbour_counts)}")

    if args.metadata_json:
        write_metadata(args.metadata_json, args, slice_meta, len(all_scores))
        print(f"Saved metadata   : {args.metadata_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
