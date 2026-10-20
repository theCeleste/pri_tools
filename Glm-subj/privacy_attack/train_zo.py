#!/usr/bin/env python3
"""Train generative ChatGLM3 LoRA with the original MeZO or DPZero idea."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import random
import time

from .common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_RAW_DATA,
    load_split,
    prepare_training_splits,
    require_empty_output,
    save_audit_splits,
    validate_chatglm_base,
    write_json,
    write_jsonl,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["mezo", "dpzero"], required=True)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--data-dir", default=str(DEFAULT_RAW_DATA))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=6750)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--zo-epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--target-modules", default="query_key_value",
        help="Comma-separated ChatGLM module names receiving Gradient LoRA",
    )
    parser.add_argument(
        "--layers-to-transform", default="",
        help="Optional comma-separated zero-based transformer layer indexes receiving Gradient LoRA",
    )
    parser.add_argument(
        "--trainable-dtype", choices=["float32", "bfloat16"], default="float32",
        help="Precision for LoRA parameters and zero-order perturbations",
    )
    parser.add_argument(
        "--loss-prefix-tokens", type=int, default=8,
        help="Use this many text tokens as context and compute causal loss only on the suffix",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-threshold", type=float, default=30.0)
    parser.add_argument("--dp-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--max-train-samples", type=int, default=0,
        help="Maximum training records; 0 uses the complete training split",
    )
    parser.add_argument(
        "--exposure-samples", type=int, default=0,
        help=(
            "Number of eligible training members to place in a deterministic high-exposure "
            "audit cohort; 0 disables the exposure stress test"
        ),
    )
    parser.add_argument(
        "--exposure-repeats", type=int, default=1,
        help="Total copies of every exposure-cohort member in the training pool",
    )
    parser.add_argument(
        "--exposure-min-suffix-tokens", type=int, default=8,
        help="Minimum attackable suffix length required for an exposure-cohort member",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="Stop after this many evaluations without an improved monitored loss; 0 disables",
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float, default=0.0,
        help="Minimum monitored-loss decrease required to reset early-stopping patience",
    )
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=450)
    parser.add_argument(
        "--checkpoint-steps", type=int, default=450,
        help="Save a recoverable adapter every N steps; 0 disables periodic checkpoints",
    )
    parser.add_argument(
        "--initial-adapter",
        help="Optional compatible checkpoint adapter used to continue an interrupted run",
    )
    parser.add_argument(
        "--accounting-previous-steps", type=int, default=0,
        help="DPZero steps already spent by initial-adapter, for cumulative privacy accounting",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args, parser):
    positive = ("rank", "lora_alpha", "max_steps", "batch_size", "max_length",
                "eval_samples", "eval_steps")
    if any(getattr(args, key) < 1 for key in positive):
        parser.error("counts and dimensions must be positive")
    if args.max_train_samples < 0:
        parser.error("max-train-samples must be nonnegative (0 means the full training split)")
    if args.exposure_samples < 0 or args.exposure_repeats < 1 or args.exposure_min_suffix_tokens < 1:
        parser.error("invalid exposure cohort configuration")
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0:
        parser.error("invalid early-stopping configuration")
    if args.checkpoint_steps < 0 or args.accounting_previous_steps < 0:
        parser.error("checkpoint-steps and accounting-previous-steps must be nonnegative")
    if args.accounting_previous_steps and not args.initial_adapter:
        parser.error("accounting-previous-steps requires initial-adapter")
    if args.exposure_samples and not args.loss_prefix_tokens:
        parser.error("exposure stress testing requires a positive loss-prefix-tokens value")
    if args.loss_prefix_tokens < 0:
        parser.error("loss-prefix-tokens must be nonnegative")
    if args.learning_rate <= 0 or args.zo_epsilon <= 0 or not 0 <= args.lora_dropout < 1:
        parser.error("invalid MeZO/LoRA parameters")
    target_modules = [value.strip() for value in args.target_modules.split(",") if value.strip()]
    if not target_modules:
        parser.error("target-modules must contain at least one module name")
    try:
        layer_indexes = [int(value.strip()) for value in args.layers_to_transform.split(",") if value.strip()]
    except ValueError:
        parser.error("layers-to-transform must be a comma-separated list of integers")
    if len(layer_indexes) != len(set(layer_indexes)) or any(index < 0 for index in layer_indexes):
        parser.error("layers-to-transform must contain unique nonnegative indexes")
    if args.method == "dpzero" and (
        args.clip_threshold <= 0 or args.dp_epsilon <= 0 or not 0 < args.dp_delta < 1
    ):
        parser.error("invalid DPZero privacy parameters")


def run(args):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from tqdm import tqdm
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("MeZO/DPZero training requires CUDA")
    if args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"gpu-id {args.gpu_id} is unavailable")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected GPU does not support bfloat16")
    dtype = getattr(torch, args.dtype)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = require_empty_output(args.output_dir)
    train_rows, validation_rows, test_rows = prepare_training_splits(
        args.data_dir, args.max_train_samples, args.seed,
    )
    audit_dir, _ = save_audit_splits(output_dir, train_rows, validation_rows, test_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    layer_indexes = [int(value.strip()) for value in args.layers_to_transform.split(",") if value.strip()]
    number_of_layers = int(getattr(config, "num_layers", 0))
    if layer_indexes and (not number_of_layers or max(layer_indexes) >= number_of_layers):
        raise ValueError(
            f"layers-to-transform must be in [0, {number_of_layers - 1}], got {layer_indexes}"
        )
    config.torch_dtype = dtype
    config.seq_length = args.max_length
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, config=config, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    target_modules = [value.strip() for value in args.target_modules.split(",") if value.strip()]
    lora_arguments = {
        "task_type": TaskType.CAUSAL_LM,
        "r": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
    }
    if layer_indexes:
        lora_arguments.update(layers_to_transform=layer_indexes, layers_pattern="layers")
    if args.initial_adapter:
        model = PeftModel.from_pretrained(model, args.initial_adapter, is_trainable=True).to(device)
    else:
        model = get_peft_model(model, LoraConfig(**lora_arguments)).to(device)
    model.eval()
    trainable_dtype = getattr(torch, args.trainable_dtype)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(dtype=trainable_dtype)
    model.print_trainable_parameters()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters")
    trainable_parameter_count = sum(parameter.numel() for _name, parameter in trainable)
    trainable_parameter_dtypes = sorted({str(parameter.dtype) for _name, parameter in trainable})
    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    minimum_sequence_length = (
        special_token_count + args.loss_prefix_tokens + args.exposure_min_suffix_tokens
    )
    if args.max_length < minimum_sequence_length:
        raise ValueError(
            f"max-length must be at least {minimum_sequence_length}: "
            f"{special_token_count} special + {args.loss_prefix_tokens} prefix + "
            f"{args.exposure_min_suffix_tokens} suffix tokens"
        )

    def text_token_count(row):
        token_ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return len(token_ids)

    required_evaluation_tokens = args.loss_prefix_tokens + args.exposure_min_suffix_tokens

    def comparable_rows(rows):
        return [row for row in rows if text_token_count(row) >= required_evaluation_tokens]

    # A high-exposure cohort is an explicit memorization stress test. Every
    # unique Subj member remains in the pool; selected originals receive extra
    # occurrences and are saved separately so the attack evaluates exactly the
    # records whose exposure was increased.
    training_rows = list(train_rows)
    exposure_audit_dir = None
    exposure_rows = []
    exposure_source_indices = []
    if args.exposure_samples:
        eligible = comparable_rows(train_rows)
        random.Random(args.seed).shuffle(eligible)
        if len(eligible) < args.exposure_samples:
            raise ValueError(
                f"Only {len(eligible)} training rows have at least {required_evaluation_tokens} "
                "text tokens; "
                f"requested {args.exposure_samples} exposure members"
            )
        exposure_rows = eligible[:args.exposure_samples]
        for _ in range(args.exposure_repeats - 1):
            training_rows.extend(exposure_rows)
        exposure_source_indices = [int(row["_source_index"]) for row in exposure_rows]
        exposure_audit_dir = output_dir / "audit_exposure_data"
        exposure_audit_dir.mkdir()
        for split, rows in (
            ("train", exposure_rows),
            ("validation", validation_rows),
            ("test", test_rows),
        ):
            write_jsonl(exposure_audit_dir / f"{split}.jsonl", rows)
        write_json(exposure_audit_dir / "manifest.json", {
            "purpose": "high_exposure_member_extraction_stress_test",
            "source_split": "train",
            "unique_members": len(exposure_rows),
            "copies_per_member": args.exposure_repeats,
            "loss_prefix_tokens": args.loss_prefix_tokens,
            "minimum_suffix_tokens": args.exposure_min_suffix_tokens,
            "source_indices": exposure_source_indices,
            "privacy_warning": (
                "Repeated occurrences are not a unique-record DP unit. The DP accountant in this run "
                "describes occurrence-level sampling, not user-level protection for the repeated member."
            ),
        })

    evaluation_train_rows = comparable_rows(train_rows)
    evaluation_validation_rows = comparable_rows(validation_rows)
    if not evaluation_train_rows or not evaluation_validation_rows:
        raise ValueError(
            "No comparable train/validation rows satisfy the configured prefix and suffix lengths"
        )

    effective_batch_size = min(args.batch_size, len(training_rows))
    print(json.dumps({
        "training_data": str(args.data_dir),
        "unique_train_records": len(train_rows),
        "training_pool_records": len(training_rows),
        "selection": "full_training_split" if args.max_train_samples == 0 else "seeded_subset",
        "exposure_members": len(exposure_source_indices),
        "exposure_copies_per_member": args.exposure_repeats if exposure_source_indices else 1,
        "comparable_train_records": len(evaluation_train_rows),
        "comparable_validation_records": len(evaluation_validation_rows),
        "required_evaluation_text_tokens": required_evaluation_tokens,
        "special_token_count": special_token_count,
        "minimum_sequence_length": minimum_sequence_length,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "approximate_unique_data_passes": args.max_steps * effective_batch_size / len(train_rows),
        "approximate_pool_passes": args.max_steps * effective_batch_size / len(training_rows),
    }, ensure_ascii=False), flush=True)

    def encode(rows):
        batch = tokenizer([row["text"] for row in rows], return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_length)
        prefix_lengths = []
        for row in rows:
            prefix = row.get("loss_prefix")
            if prefix is None:
                prefix_lengths.append(args.loss_prefix_tokens)
                continue
            if not isinstance(prefix, str) or not prefix.strip():
                raise ValueError("loss_prefix must be a non-empty string")
            ids = tokenizer(prefix.strip(), add_special_tokens=False)["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            prefix_lengths.append(len(ids))
        batch["loss_prefix_lengths"] = torch.tensor(prefix_lengths, dtype=torch.long)
        return batch

    def per_example_loss(batch):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prefix_lengths = batch["loss_prefix_lengths"].to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        shift_logits = logits[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()
        # A transition is valid only when both tokens are non-padding. ChatGLM
        # uses left-padded batches, so masking only the target includes one
        # artificial pad-to-first-token prediction per sequence.
        mask = (attention_mask[:, :-1] * attention_mask[:, 1:]).float()
        if bool(prefix_lengths.max()):
            context_targets = special_token_count - 1 + prefix_lengths
            mask = mask * (mask.cumsum(dim=1) > context_targets.unsqueeze(1))
        losses = functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]), targets.view(-1), reduction="none"
        ).view(targets.shape)
        return (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    @torch.no_grad()
    def sample_direction(seed):
        generator = torch.Generator(device=device).manual_seed(seed)
        return [
            (
                name,
                parameter,
                torch.randn(parameter.shape, generator=generator, device=device, dtype=parameter.dtype),
            )
            for name, parameter in trainable
        ]

    @torch.no_grad()
    def perturb(direction_items, scale):
        for _name, parameter, direction in direction_items:
            parameter.add_(direction, alpha=scale * args.zo_epsilon)

    @torch.no_grad()
    def update(direction_items, projected_gradient):
        for name, parameter, direction in direction_items:
            update_value = direction * projected_gradient
            if args.weight_decay and not any(key in name.lower() for key in ("bias", "norm")):
                update_value = update_value + args.weight_decay * parameter
            parameter.add_(update_value, alpha=-args.learning_rate)

    sample_rate = effective_batch_size / len(training_rows)
    noise_multiplier = None
    noise_std = None
    actual_epsilon = None
    if args.method == "dpzero":
        try:
            from opacus.accountants import RDPAccountant
            from opacus.accountants.utils import get_noise_multiplier
        except ImportError as error:
            raise RuntimeError("DPZero accounting requires opacus==1.5.4") from error
        accounting_steps = args.accounting_previous_steps + args.max_steps
        noise_multiplier = float(get_noise_multiplier(
            target_epsilon=args.dp_epsilon,
            target_delta=args.dp_delta,
            sample_rate=sample_rate,
            steps=accounting_steps,
            accountant="rdp",
        ))
        noise_std = 2 * noise_multiplier * args.clip_threshold / effective_batch_size
        accountant = RDPAccountant()
        for _ in range(accounting_steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        actual_epsilon = float(accountant.get_epsilon(delta=args.dp_delta))

    @torch.inference_mode()
    def evaluate(rows):
        values = []
        for start in range(0, min(len(rows), args.eval_samples), args.batch_size):
            values.extend(float(value) for value in per_example_loss(encode(rows[start:start + args.batch_size])).cpu())
        mean_loss = sum(values) / len(values)
        return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20))}

    # Continued runs must not replay the exact data ordering, perturbation
    # directions, or DP noise used before the checkpoint.
    resume_offset = args.accounting_previous_steps
    order_rng = random.Random(args.seed + 1 + resume_offset)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 2 + resume_offset)
    order = []
    cursor = 0

    def next_rows():
        nonlocal order, cursor
        if cursor + args.batch_size > len(order):
            order = list(range(len(training_rows)))
            order_rng.shuffle(order)
            cursor = 0
        indexes = order[cursor:cursor + args.batch_size]
        cursor += len(indexes)
        return [training_rows[index] for index in indexes]

    initial_train_metrics = evaluate(evaluation_train_rows)
    initial_validation_metrics = evaluate(evaluation_validation_rows)
    initial_record = {
        "step": 0,
        "online_loss": None,
        "online_loss_rolling_mean": None,
        "projected_gradient": None,
        "train": initial_train_metrics,
        "validation": initial_validation_metrics,
        "generalization_gap": initial_validation_metrics["loss"] - initial_train_metrics["loss"],
    }
    if exposure_rows:
        initial_record["exposure"] = evaluate(exposure_rows)
        initial_record["exposure_advantage"] = (
            initial_validation_metrics["loss"] - initial_record["exposure"]["loss"]
        )
    best_monitored_loss = (
        initial_record["exposure"]["loss"] if exposure_rows else initial_record["train"]["loss"]
    )
    early_stopping_reference_loss = best_monitored_loss
    best_step = 0
    best_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainable
    }
    initial_record.update({
        "monitored_loss": best_monitored_loss,
        "best_step": best_step,
        "best_monitored_loss": best_monitored_loss,
        "evaluations_without_improvement": 0,
    })
    history = [initial_record]
    print(json.dumps(initial_record, ensure_ascii=False), flush=True)
    evaluations_without_improvement = 0
    stopped_early = False
    completed_steps = 0
    started = time.time()
    rolling_losses = deque(maxlen=100)
    checkpoints = []

    def save_checkpoint(step):
        checkpoint_dir = output_dir / "checkpoints" / f"step-{step:08d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(checkpoint_dir)
        write_json(checkpoint_dir / "checkpoint_metadata.json", {
            "method": args.method,
            "completed_steps_this_run": step,
            "accounting_previous_steps": args.accounting_previous_steps,
            "cumulative_steps": args.accounting_previous_steps + step,
            "data_dir": str(args.data_dir),
            "recover_with": (
                "Pass this directory as --initial-adapter, use a new empty --output-dir, and set "
                f"--accounting-previous-steps {args.accounting_previous_steps + step}."
            ),
        })
        checkpoints.append(str(checkpoint_dir.resolve()))
    progress = tqdm(range(1, args.max_steps + 1), desc=args.method)
    for step in progress:
        completed_steps = step
        batch = encode(next_rows())
        direction_seed = args.seed * 1_000_003 + resume_offset + step
        direction_items = sample_direction(direction_seed)
        perturb(direction_items, 1.0)
        loss_plus = per_example_loss(batch)
        perturb(direction_items, -2.0)
        loss_minus = per_example_loss(batch)
        perturb(direction_items, 1.0)
        projected_per_example = (loss_plus - loss_minus) / (2 * args.zo_epsilon)
        if args.method == "dpzero":
            projected = projected_per_example.clamp(-args.clip_threshold, args.clip_threshold).mean()
            projected = projected + torch.randn((), generator=noise_generator, device=device) * noise_std
        else:
            projected = projected_per_example.mean()
        update(direction_items, projected)
        online_loss = float(loss_plus.mean().cpu())
        rolling_losses.append(online_loss)
        rolling_loss = sum(rolling_losses) / len(rolling_losses)
        progress.set_postfix(loss=f"{online_loss:.4f}", avg100=f"{rolling_loss:.4f}")
        if args.checkpoint_steps and step % args.checkpoint_steps == 0:
            save_checkpoint(step)
        if step % args.eval_steps == 0 or step == args.max_steps:
            train_metrics = evaluate(evaluation_train_rows)
            validation_metrics = evaluate(evaluation_validation_rows)
            record = {
                "step": step,
                "online_loss": online_loss,
                "online_loss_rolling_mean": rolling_loss,
                "projected_gradient": float(projected.cpu()),
                "train": train_metrics,
                "validation": validation_metrics,
                "generalization_gap": validation_metrics["loss"] - train_metrics["loss"],
            }
            if exposure_rows:
                record["exposure"] = evaluate(exposure_rows)
                record["exposure_advantage"] = (
                    validation_metrics["loss"] - record["exposure"]["loss"]
                )
            monitored_loss = (
                record["exposure"]["loss"] if exposure_rows else record["train"]["loss"]
            )
            if monitored_loss < best_monitored_loss:
                best_monitored_loss = monitored_loss
                best_step = step
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in trainable
                }

            significant_improvement = (
                monitored_loss
                < early_stopping_reference_loss - args.early_stopping_min_delta
            )
            if significant_improvement:
                early_stopping_reference_loss = monitored_loss
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            record["monitored_loss"] = monitored_loss
            record["best_step"] = best_step
            record["best_monitored_loss"] = best_monitored_loss
            record["evaluations_without_improvement"] = evaluations_without_improvement
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if (
                args.early_stopping_patience
                and evaluations_without_improvement >= args.early_stopping_patience
            ):
                stopped_early = True
                break

    if best_state is None:
        raise RuntimeError("Training completed without producing an evaluation checkpoint")
    with torch.no_grad():
        for name, parameter in trainable:
            parameter.copy_(best_state[name].to(device=parameter.device, dtype=parameter.dtype))

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    privacy = None
    if args.method == "dpzero":
        completed_accountant = RDPAccountant()
        for _ in range(args.accounting_previous_steps + completed_steps):
            completed_accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        actual_epsilon = float(completed_accountant.get_epsilon(delta=args.dp_delta))
        privacy = {
            "mechanism": "DPZero projected-gradient clipping and Gaussian noise",
            "target_epsilon": args.dp_epsilon,
            "actual_epsilon": actual_epsilon,
            "delta": args.dp_delta,
            "noise_multiplier": noise_multiplier,
            "projected_gradient_noise_std": noise_std,
            "clip_threshold": args.clip_threshold,
            "sample_rate": sample_rate,
            "accounting_unit": "training-pool occurrence",
            "accounted_steps": args.accounting_previous_steps + completed_steps,
            "accounting_assumption": "RDP accounting using the imported DPZero calibration formula",
            "unique_record_warning": (
                "When exposure_repeats is greater than one, epsilon is not a unique-record or user-level "
                "guarantee for a repeated member."
            ),
        }
    report = {
        "purpose": f"{args.method}_generative_causal_lm",
        "method": args.method,
        "protected": args.method == "dpzero",
        "adapter_task": "CAUSAL_LM",
        "arguments": vars(args),
        "completed_steps": completed_steps,
        "accounting_previous_steps": args.accounting_previous_steps,
        "cumulative_steps": args.accounting_previous_steps + completed_steps,
        "stopped_early": stopped_early,
        "best_step": best_step,
        "best_monitored_loss": best_monitored_loss,
        "train_records": len(train_rows),
        "training_pool_records": len(training_rows),
        "approximate_data_passes": completed_steps * effective_batch_size / len(train_rows),
        "approximate_pool_passes": completed_steps * effective_batch_size / len(training_rows),
        "target_modules": target_modules,
        "layers_to_transform": layer_indexes,
        "loss_prefix_tokens": args.loss_prefix_tokens,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_dtypes": trainable_parameter_dtypes,
        "audit_data_dir": str(audit_dir.resolve()),
        "exposure_audit_data_dir": (
            str(exposure_audit_dir.resolve()) if exposure_audit_dir is not None else None
        ),
        "exposure_source_indices": exposure_source_indices,
        "privacy": privacy,
        "checkpoints": checkpoints,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    write_json(output_dir / "training_report.json", report)
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    validate_chatglm_base(args.base_model)
    if args.initial_adapter:
        from .common import validate_adapter
        validate_adapter(args.initial_adapter)
    for split in ("train", "validation", "test"):
        load_split(args.data_dir, split)
    if args.dry_run:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return vars(args)
    return run(args)


if __name__ == "__main__":
    main()
