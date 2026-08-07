from utils import * 

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json 
import random
# random.seed(42)
from torch.distributions import Laplace
from datetime import datetime
import math
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase as Tokenizer
from transformers import PreTrainedModel as Model
from dataclasses import dataclass
from typing import Iterable
from steering_vectors import train_steering_vector, pca_aggregator
from steering_wrapper import FrozenSteering
import matplotlib.pyplot as plt
import argparse 

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

MWEData = list[dict[str, str]]

logfile = "experiment_logs.jsonl"

def log_experiment(
    logfile: str,
    steering_type: str,
    multiplier: float,
    result: dict,
    model_name: str,
    dataset: str,
    layers,
    noise_multiplier=None,
    clip=None,
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
        pca_steering = train_steering_vector(
            model,
            tokenizer,
            train_dataset,
            read_token_index=-2,
            show_progress=True,
            aggregator=pca_aggregator(),
            layers=layers
        )
        print("PCA Steering")
        for multiplier in (-2, 2):
            with pca_steering.apply(model, multiplier=multiplier, min_token_index=0):
                result = evaluate_model(model, tokenizer, test_dataset)
                print(f"{multiplier=} | {result=}")
    
    mean_steering = train_steering_vector(
        model, 
        tokenizer,
        train_dataset,
        read_token_index=-2,
        show_progress=True,
        aggregator=scaled_mean,
        layers=layers
    )

    torch.save(
        {
            "layer_activations": {
                k: v.detach().cpu()       # 一定转到 CPU
                for k, v in mean_steering.layer_activations.items()
            },
            "layer_type": mean_steering.layer_type,
        },
        "mean_steering_vectors_llama_7b_hf.pt"
    )

    private_steering = train_steering_vector(
        model,
        tokenizer,
        train_dataset,
        read_token_index=-2,
        show_progress=True,
        aggregator=priv_mean,
        layers=layers
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
        "private_steering_vectors_llama_7b_hf.pt"
    )

    # ckpt = torch.load("mean_steering_vectors_chat_7b.pt", map_location="cpu")
    # mean_steering = FrozenSteering(ckpt["layer_activations"])

    # ckpt = torch.load("private_steering_vectors_chat_7b.pt", map_location="cpu")
    # private_steering = FrozenSteering(ckpt["layer_activations"])


    # private_steering = torch.load("private_steering.pkl")

    print("Mean Steering")
    for multiplier in (-2, 0, 2):
        with mean_steering.apply(model, multiplier=multiplier, min_token_index=0):
            result = evaluate_model(model, tokenizer, test_dataset)
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
            )

    print("Private Mean Steering")
    for multiplier in (-2, 2):
        with private_steering.apply(model, multiplier=multiplier, min_token_index=0):
            result = evaluate_model(model, tokenizer, test_dataset)
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