from utils import * 

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json 
import random
from torch.distributions import Laplace
from datetime import datetime
import math
import torch
from tqdm import tqdm
import time
import resource
from transformers import PreTrainedTokenizerBase as Tokenizer
from transformers import PreTrainedModel as Model
from dataclasses import dataclass
from typing import Any, Callable, Iterable
from steering_vectors import train_steering_vector, pca_aggregator
from steering_wrapper import FrozenSteering
import matplotlib.pyplot as plt
import argparse 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MWEData = list[dict[str, str]]

logfile = "experiment_logs.jsonl"


def _to_mb(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    return num_bytes / (1024 ** 2)


def get_peak_cpu_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KB.
    return int(usage.ru_maxrss * 1024)


def measure_stage(
    stage_name: str,
    fn: Callable[[], Any],
    num_samples: int | None = None,
):
    cuda_device = torch.device(device) if torch.cuda.is_available() else None
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)

    start_time = time.perf_counter()
    result = fn()

    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(cuda_device))
    else:
        peak_gpu_memory_bytes = None

    elapsed_seconds = time.perf_counter() - start_time
    throughput = None
    if num_samples is not None and elapsed_seconds > 0:
        throughput = num_samples / elapsed_seconds
    peak_cpu_rss_bytes = get_peak_cpu_rss_bytes()

    metrics = {
        "stage": stage_name,
        "num_samples": num_samples,
        "elapsed_seconds": elapsed_seconds,
        "throughput_samples_per_second": throughput,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "peak_gpu_memory_mb": _to_mb(peak_gpu_memory_bytes),
        "peak_cpu_rss_bytes": peak_cpu_rss_bytes,
        "peak_cpu_rss_mb": _to_mb(peak_cpu_rss_bytes),
    }

    metrics_str = [
        f"elapsed={elapsed_seconds:.4f}s",
        f"peak_cpu={metrics['peak_cpu_rss_mb']:.2f}MB",
    ]
    if metrics["peak_gpu_memory_mb"] is not None:
        metrics_str.append(f"peak_gpu={metrics['peak_gpu_memory_mb']:.2f}MB")
    if throughput is not None:
        metrics_str.append(f"throughput={throughput:.4f} samples/s")
    print(f"[{stage_name}] " + " | ".join(metrics_str))

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
    train_data: list[MWEData] = json.load(open(f"./datasets/generate/{dataset}/generate_dataset.json", 'r'))
    test_data: list[MWEData] = json.load(open(f"./datasets/test/{dataset}/test_dataset_ab.json", 'r'))

    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(test_data)
    train_data = train_data
    test_data = test_data

    train_dataset = make_dataset(train_data, tokenizer)
    test_dataset = make_dataset(test_data, tokenizer)


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
            num_samples=len(train_dataset),
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
        print("PCA Steering")
        for multiplier in (-2, 2):
            with pca_steering.apply(model, multiplier=multiplier, min_token_index=0):
                result, eval_metrics = measure_stage(
                    stage_name=f"pca_eval_multiplier_{multiplier}",
                    num_samples=len(test_dataset),
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
    
    mean_steering, mean_train_metrics = measure_stage(
        stage_name="mean_train",
        num_samples=len(train_dataset),
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
        num_samples=len(train_dataset),
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
                num_samples=len(test_dataset),
                fn=lambda: evaluate_model(model, tokenizer, test_dataset)
            )
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

    print("Private Mean Steering")
    for multiplier in (-2, 2):
        with private_steering.apply(model, multiplier=multiplier, min_token_index=0):
            result, eval_metrics = measure_stage(
                stage_name=f"private_mean_eval_multiplier_{multiplier}",
                num_samples=len(test_dataset),
                fn=lambda: evaluate_model(model, tokenizer, test_dataset)
            )
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
            if prompt:
                generate_text(prompt, model, tokenizer)


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
    pca=False
)
