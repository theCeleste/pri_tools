from utils import * 

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json 
import random
from torch.distributions import Laplace
from datetime import datetime
import math
from tqdm import tqdm
import time
import threading
import psutil
from transformers import PreTrainedTokenizerBase as Tokenizer
from transformers import PreTrainedModel as Model
from dataclasses import dataclass
from typing import Any, Callable, Iterable
from steering_vectors import train_steering_vector, pca_aggregator
from steering_wrapper import FrozenSteering
import matplotlib.pyplot as plt
import argparse 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

MWEData = list[dict[str, str]]

logfile = "experiment_logs.jsonl"
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.05
PROGRAM_START_TIME = time.perf_counter()
PROFILED_STAGE_NAMES = {
    "mean_train",
    "private_mean_train",
    "mean_eval_multiplier_0",
    "private_mean_eval_multiplier_2",
}


def _to_mb(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    return num_bytes / (1024 ** 2)


def _max_or_none(values: Iterable[int | float | None]) -> int | float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return max(valid_values)


def summarize_profile_metrics(stage_metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not stage_metrics:
        return None

    total_elapsed_seconds = time.perf_counter() - PROGRAM_START_TIME
    total_num_samples = sum(metric["num_samples"] or 0 for metric in stage_metrics)
    throughput = _max_or_none(
        metric["throughput_samples_per_second"] for metric in stage_metrics
    )

    peak_cpu_memory_rss_bytes = _max_or_none(
        metric["peak_cpu_memory_rss_bytes"] for metric in stage_metrics
    )
    peak_gpu_memory_allocated_bytes = _max_or_none(
        metric["peak_gpu_memory_allocated_bytes"] for metric in stage_metrics
    )
    peak_gpu_memory_reserved_bytes = _max_or_none(
        metric["peak_gpu_memory_reserved_bytes"] for metric in stage_metrics
    )

    return {
        "profiled_stages": [metric["stage"] for metric in stage_metrics],
        "num_profiled_stages": len(stage_metrics),
        "total_num_samples": total_num_samples,
        "total_elapsed_seconds": total_elapsed_seconds,
        "throughput_samples_per_second": throughput,
        "peak_cpu_memory_rss_bytes": peak_cpu_memory_rss_bytes,
        "peak_cpu_memory_rss_mb": _to_mb(peak_cpu_memory_rss_bytes),
        "peak_gpu_memory_allocated_bytes": peak_gpu_memory_allocated_bytes,
        "peak_gpu_memory_allocated_mb": _to_mb(peak_gpu_memory_allocated_bytes),
        "peak_gpu_memory_reserved_bytes": peak_gpu_memory_reserved_bytes,
        "peak_gpu_memory_reserved_mb": _to_mb(peak_gpu_memory_reserved_bytes),
    }


def print_profile_summary(stage_metrics: list[dict[str, Any]]) -> None:
    summary = summarize_profile_metrics(stage_metrics)
    if summary is None:
        return

    summary_parts = [
        f"stages={summary['num_profiled_stages']}",
        f"elapsed_program={summary['total_elapsed_seconds']:.4f}s",
        f"peak_cpu={summary['peak_cpu_memory_rss_mb']:.2f}MB",
    ]
    if summary["peak_gpu_memory_allocated_mb"] is not None:
        summary_parts.append(
            f"peak_gpu_allocated={summary['peak_gpu_memory_allocated_mb']:.2f}MB"
        )
        summary_parts.append(
            f"peak_gpu_reserved={summary['peak_gpu_memory_reserved_mb']:.2f}MB"
        )
    if summary["throughput_samples_per_second"] is not None:
        summary_parts.append(
            f"throughput={summary['throughput_samples_per_second']:.4f} samples/s"
        )

    print("[profile_summary] " + " | ".join(summary_parts))


class PeakMemoryMonitor:
    def __init__(self, sample_interval_seconds: float = MEMORY_SAMPLE_INTERVAL_SECONDS):
        self.sample_interval_seconds = sample_interval_seconds
        self.process = psutil.Process()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_rss_bytes: int | None = None
        self.end_rss_bytes: int | None = None
        self.peak_rss_bytes: int | None = None

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self.sample_once()

    def sample_once(self) -> None:
        rss_bytes = self.process.memory_info().rss
        if self.peak_rss_bytes is None or rss_bytes > self.peak_rss_bytes:
            self.peak_rss_bytes = rss_bytes

    def start(self) -> None:
        self.start_rss_bytes = self.process.memory_info().rss
        self.peak_rss_bytes = self.start_rss_bytes
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.end_rss_bytes = self.process.memory_info().rss
        self.sample_once()


def measure_stage(
    stage_name: str,
    fn: Callable[[], Any],
    num_samples: int | None = None,
):
    if stage_name not in PROFILED_STAGE_NAMES:
        return fn(), None

    cuda_device = torch.device(device) if torch.cuda.is_available() else None
    memory_monitor = PeakMemoryMonitor()
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)

    start_time = time.perf_counter()
    memory_monitor.start()
    try:
        result = fn()
    finally:
        memory_monitor.stop()

    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        peak_gpu_memory_allocated_bytes = int(torch.cuda.max_memory_allocated(cuda_device))
        peak_gpu_memory_reserved_bytes = int(torch.cuda.max_memory_reserved(cuda_device))
    else:
        peak_gpu_memory_allocated_bytes = None
        peak_gpu_memory_reserved_bytes = None

    elapsed_seconds = time.perf_counter() - start_time
    throughput = None
    if num_samples is not None and elapsed_seconds > 0:
        throughput = num_samples / elapsed_seconds

    metrics = {
        "stage": stage_name,
        "num_samples": num_samples,
        "elapsed_seconds": elapsed_seconds,
        "throughput_samples_per_second": throughput,
        "start_cpu_memory_rss_bytes": memory_monitor.start_rss_bytes,
        "start_cpu_memory_rss_mb": _to_mb(memory_monitor.start_rss_bytes),
        "end_cpu_memory_rss_bytes": memory_monitor.end_rss_bytes,
        "end_cpu_memory_rss_mb": _to_mb(memory_monitor.end_rss_bytes),
        "peak_cpu_memory_rss_bytes": memory_monitor.peak_rss_bytes,
        "peak_cpu_memory_rss_mb": _to_mb(memory_monitor.peak_rss_bytes),
        "peak_gpu_memory_allocated_bytes": peak_gpu_memory_allocated_bytes,
        "peak_gpu_memory_allocated_mb": _to_mb(peak_gpu_memory_allocated_bytes),
        "peak_gpu_memory_reserved_bytes": peak_gpu_memory_reserved_bytes,
        "peak_gpu_memory_reserved_mb": _to_mb(peak_gpu_memory_reserved_bytes),
    }

    return result, metrics


def log_experiment(
    logfile: str,
    steering_type: str,
    multiplier: float | None,
    result,
    model_name: str,
    dataset: str,
    layers,
    noise_multiplier=None,
    clip=None,
    stage: str | None = None,
    metrics: dict | None = None,
):
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "steering_type": steering_type,
        "multiplier": multiplier,
        "result": result,
        "model": model_name,
        "dataset": dataset,
        "layers": layers,
        "noise_multiplier": noise_multiplier,
        "clip": clip,
        "stage": stage,
        "metrics": metrics,
    }

    with open(logfile, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_results(dataset, prompt=None, pca=False, layers=[11,12,13,14,15], noise_multiplier=0.01, clip=20):
    train_data: list[MWEData] = json.load(open(r"/home/zjs/code/x73/x73-privacy-tool/PrivSV/datasets/generate" + f"/{dataset}/generate_dataset.json", 'r'))
    test_data: list[MWEData] = json.load(open(r"/home/zjs/code/x73/x73-privacy-tool/PrivSV/datasets/test" + f"/{dataset}/test_dataset_ab.json", 'r'))

    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(test_data)
    train_data = train_data
    test_data = test_data

    train_dataset = make_dataset(train_data, tokenizer)
    test_dataset = make_dataset(test_data, tokenizer)
    profiled_stage_metrics: list[dict[str, Any]] = []


    def scaled_mean(pos, neg):
        diff = pos - neg
        scale = torch.max(torch.norm(diff, dim=1))
        diff /= scale
        return torch.mean(diff, dim=0)

    
    def priv_mean(pos, neg):
        diff = pos - neg
        C = clip
        norms = torch.norm(diff, dim=1)
        scale_factors = torch.clamp(C/norms, max=1.0).view(-1, 1)
        diff = diff * scale_factors
        diff /= C
        mu = torch.mean(diff, dim=0)
        noise = torch.normal(0, torch.tensor(1.), size=mu.shape).to(device)
        return mu + (noise_multiplier * noise)

    if pca:
        pca_steering, pca_train_metrics = measure_stage(
            stage_name="pca_train",
            num_samples=len(train_dataset)*2,
            fn=lambda: train_steering_vector(
                model,
                tokenizer,
                train_dataset,
                read_token_index=-2,
                show_progress=True,
                aggregator=pca_aggregator(),
                batch_size=4,
                layers=layers
            ),
        )
        # print(num_sampless)
        log_experiment(
            logfile=logfile,
            steering_type="pca",
            multiplier=None,
            result=None,
            model_name=model_name,
            dataset=args.dataset,
            layers=layers,
            stage="train",
            metrics=pca_train_metrics,
        )
        if pca_train_metrics is not None:
            profiled_stage_metrics.append(pca_train_metrics)
        print("PCA Steering")
        for multiplier in (-2, 2):
            with pca_steering.apply(model, multiplier=multiplier, min_token_index=0):
                result, eval_metrics = measure_stage(
                    stage_name=f"pca_eval_multiplier_{multiplier}",
                    num_samples=len(test_dataset)*2,
                    fn=lambda: evaluate_model(model, tokenizer, test_dataset)
                )
                print(f"{multiplier=} | {result=}")
                log_experiment(
                    logfile=logfile,
                    steering_type="pca",
                    multiplier=multiplier,
                    result=result,
                    model_name=model_name,
                    dataset=args.dataset,
                    layers=layers,
                    stage="eval",
                    metrics=eval_metrics,
                )
                if eval_metrics is not None:
                    profiled_stage_metrics.append(eval_metrics)
    
    mean_steering, mean_train_metrics = measure_stage(
        stage_name="mean_train",
        num_samples=len(train_dataset)*2,
        fn=lambda: train_steering_vector(
            model, 
            tokenizer,
            train_dataset,
            read_token_index=-2,
            show_progress=True,
            aggregator=scaled_mean,
            batch_size=4,
            layers=layers
        ),
    )
    # print(len(train_dataset))
    log_experiment(
        logfile=logfile,
        steering_type="mean",
        multiplier=None,
        result=None,
        model_name=model_name,
        dataset=args.dataset,
        layers=layers,
        stage="train",
        metrics=mean_train_metrics,
    )
    if mean_train_metrics is not None:
        profiled_stage_metrics.append(mean_train_metrics)

    torch.save(
        {
            "layer_activations": {
                k: v.detach().cpu()       # 一定转到 CPU
                for k, v in mean_steering.layer_activations.items()
            },
            "layer_type": mean_steering.layer_type,
        },
        "mean_steering_vectors_llama_7b_hf-m.pt"
    )

    private_steering, private_train_metrics = measure_stage(
        stage_name="private_mean_train",
        num_samples=len(train_dataset)*2,
        fn=lambda: train_steering_vector(
            model,
            tokenizer,
            train_dataset,
            read_token_index=-2,
            show_progress=True,
            aggregator=priv_mean,
            batch_size=4,
            layers=layers
        ),
    )
    # print(len(train_dataset))
    log_experiment(
        logfile=logfile,
        steering_type="private_mean",
        multiplier=None,
        result=None,
        model_name=model_name,
        dataset=args.dataset,
        layers=layers,
        noise_multiplier=args.noise_multiplier,
        clip=args.clip,
        stage="train",
        metrics=private_train_metrics,
    )
    if private_train_metrics is not None:
        profiled_stage_metrics.append(private_train_metrics)

    # print(private_steering)
    # print(dir(private_steering))

    torch.save(
        {
            "layer_activations": {
                k: v.detach().cpu()       # 一定转到 CPU
                for k, v in private_steering.layer_activations.items()
            },
            "layer_type": private_steering.layer_type,
        },
        "private_steering_vectors_llama_7b_hf-m.pt"
    )

    # ckpt = torch.load("mean_steering_vectors_chat_7b.pt", map_location="cpu")
    # mean_steering = FrozenSteering(ckpt["layer_activations"])

    # ckpt = torch.load("private_steering_vectors_chat_7b.pt", map_location="cpu")
    # private_steering = FrozenSteering(ckpt["layer_activations"])


    # private_steering = torch.load("private_steering.pkl")

    print("Mean Steering")
    for multiplier in (-2, 0, 2):
        with mean_steering.apply(model, multiplier=multiplier, min_token_index=0):
            result, eval_metrics = measure_stage(
                stage_name=f"mean_eval_multiplier_{multiplier}",
                num_samples=len(test_dataset)*2,
                fn=lambda: evaluate_model(model, tokenizer, test_dataset)
            )
            # print(len(test_dataset))
            print(f"{multiplier=} | {result=}")
            # generate_text(prompt, model, tokenizer)
            log_experiment(
                logfile=logfile,
                steering_type="mean",
                multiplier=multiplier,
                result=result,
                model_name=model_name,
                dataset=args.dataset,
                layers=layers,
                stage="eval",
                metrics=eval_metrics,
            )
            if eval_metrics is not None:
                profiled_stage_metrics.append(eval_metrics)

    print("Private Mean Steering")
    for multiplier in (-2, 2):
        with private_steering.apply(model, multiplier=multiplier, min_token_index=0):
            result, eval_metrics = measure_stage(
                stage_name=f"private_mean_eval_multiplier_{multiplier}",
                num_samples=len(test_dataset)*2,
                fn=lambda: evaluate_model(model, tokenizer, test_dataset)
            )
            # print(len(test_dataset))
            print(f"{multiplier=} | {result=}")
            log_experiment(
                logfile=logfile,
                steering_type="private_mean",
                multiplier=multiplier,
                result=result,
                model_name=model_name,
                dataset=args.dataset,
                layers=layers,
                noise_multiplier=args.noise_multiplier,
                clip=args.clip,
                stage="eval",
                metrics=eval_metrics,
            )
            if eval_metrics is not None:
                profiled_stage_metrics.append(eval_metrics)
            if prompt:
                generate_text(prompt, model, tokenizer)

    print_profile_summary(profiled_stage_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7B-hf",
        help="huggingface llm identifier"
    )
    parser.add_argument(
        "--dataset",
        default="sycophancy",
        help="name of dataset",
        choices=['sycophancy', 'hallucination', 'refusal', 'myopic-reward', 'survival-instinct', 'coordinate-other-ais', 'corrigible-neutral-HHH', 'sst2']
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs='+',
        help="layers to apply psa on"
    )
    parser.add_argument(
        "--noise_multiplier",
        default=0.02,
        type=float,
        help="amount of noise added in psa"
    )
    parser.add_argument(
        "--clip",
        default=20,
        type=int,
        help="clipping factor"
    )
    args = parser.parse_args()


model_name = args.model
model, tokenizer = get_model_and_tokenizer(model_name, load_bfloat=False)

get_results(
    dataset=args.dataset,
    layers=args.layers,
    noise_multiplier=args.noise_multiplier,
    pca=False,
    clip=args.clip
)
