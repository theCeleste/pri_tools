import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import platform
import time
import src.tasks
import transformers as hf_transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
from typing import Union, Optional
import torch
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from src.tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from src.metrics import calculate_metric
from src.utils import *
from src.trainer import OurTrainer
import random
from functools import partial

@dataclass
class OurArguments(TrainingArguments):
    # Transformers 4.28 defaults to its deprecated AdamW implementation.
    # Use the maintained PyTorch optimizer for regular fine-tuning.
    optim: str = "adamw_torch"

    # dataset and sampling strategy
    task_name: str = "SST2" # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP

    # Number of examples
    num_train: int = 0 # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None # (only enabled with training) number of development samples
    num_eval: int = None # number of evaluation samples
    num_train_sets: int = None # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None # designated seed to sample training samples/demos
    result_file: str = None # file name for saving performance; if None, then use the task name, model name, and config
    experiment_suite: str = "manual"
    experiment_id: str = "manual"

    # Model loading
    model_name: str = "meta-llama/Llama-2-7b-hf" # Llama 2 base model name or local path
    load_float16: bool = False # load model parameters as float16
    load_bfloat16: bool = False # load model parameters as bfloat16
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take
    no_auto_device: bool = False # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False # whether to use SFC calibration
    icl_sfc: bool = False # whether to use SFC calibration for ICL samples

    # Training
    trainer: str = "none" 
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo: zeroth-order (MeZO) training
    only_train_option: bool = True # whether to only train the option part of the input
    train_as_classification: bool = False # take the log likelihood of all options and train as classification 

    # MeZO
    zo_eps: float = 1e-3 # eps in MeZO

    # Prefix tuning
    prefix_tuning: bool = False # whether to use prefix tuning
    num_prefix: int = 5 # number of prefixes to use
    no_reparam: bool = True # do not use reparameterization trick
    prefix_init_by_real_act: bool = True # initialize prefix by real activations of random words

    # LoRA
    lora: bool = False # whether to use LoRA
    lora_alpha: int = 16 # alpha in LoRA
    lora_r: int = 8 # r in LoRA
    lora_target_modules: str = "q_proj,v_proj" # comma-separated Llama attention projections
    lora_num_layers: int = -1 # inject into the last N layers; -1 means all layers

    # Generation
    sampling: bool = False # whether to use sampling
    temperature: float = 1.0 # temperature for generation
    num_beams: int = 1 # number of beams for generation
    top_k: int = None # top-k for generation
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of sentence token

    # Saving
    save_model: bool = False # whether to save the model
    save_adapter_only: bool = True # save only LoRA/head/prefix tensors when possible
    no_eval: bool = False # whether to skip evaluation
    dev_only: bool = False # tuning mode: evaluate dev but never touch the formal test split
    tag: str = "" # saving tag

    # Linear probing
    linear_probing: bool = False # whether to do linear probing
    lp_early_stopping: bool = False # whether to do early stopping in linear probing
    head_tuning: bool = False # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False # untie the embeddings and LM head

    # Display
    verbose: bool = False # verbose output

    # Non-diff objective
    non_diff: bool = False # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False # save model when interrupted (useful for long training)
    
    # ZODPGA args
    zo_dpga: bool = False # 'whether to use ZODPGA in training'}
    zo_dpga_clip_threshold: float = 1.0 # "ZODPGA clip threshold"
    dp_epsilon: float = 6.0
    dp_delta: float = 1e-5

    # Llama-specific safety controls
    trust_remote_code: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    if args.load_float16 and args.load_bfloat16:
        raise ValueError("Choose only one of --load_float16 and --load_bfloat16")
    if args.trainer == "zo" and args.gradient_accumulation_steps != 1:
        raise ValueError("MeZO/ZODPGA does not support gradient accumulation; set --gradient_accumulation_steps 1")
    if args.zo_dpga and args.trainer != "zo":
        raise ValueError("--zo_dpga requires --trainer zo")
    if args.zo_dpga and args.zo_dpga_clip_threshold <= 0:
        raise ValueError("--zo_dpga_clip_threshold must be positive")
    if args.zo_dpga and not (args.dp_epsilon > 0 and 0 < args.dp_delta < 1):
        raise ValueError("ZODPGA requires dp_epsilon > 0 and 0 < dp_delta < 1")
    if args.zo_dpga and not args.dataloader_drop_last:
        logger.info("ZODPGA strict privacy mode enables --dataloader_drop_last for a fixed batch size")
        args.dataloader_drop_last = True
    if args.dev_only and args.num_dev is None:
        raise ValueError("--dev_only requires --num_dev")
    if args.dev_only and args.no_eval:
        raise ValueError("--dev_only and --no_eval are mutually exclusive")
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        load_start = time.perf_counter()
        self.model, self.tokenizer = self.load_model()
        self.model_load_seconds = time.perf_counter() - load_start

    def _write_benchmark(self, trainer, train_output, wall_seconds, resumed_from_checkpoint):
        if not self.args.should_save:
            return
        mode = "prefix" if self.args.prefix_tuning else "lora" if self.args.lora else "head" if self.args.head_tuning else "full"
        method = "zo_dpga" if self.args.zo_dpga else "mezo" if self.args.trainer == "zo" else self.args.trainer
        gpu_stats = []
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(device_id)
                gpu_stats.append({
                    "device": device_id,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device_id),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device_id),
                })
        privacy = None
        if self.args.zo_dpga:
            privacy = {
                "epsilon": self.args.dp_epsilon,
                "delta": self.args.dp_delta,
                "clip_threshold": self.args.zo_dpga_clip_threshold,
                "zo_eps": self.args.zo_eps,
                "sample_rate": getattr(trainer, "zo_dpga_sample_rate", None),
                "noise_multiplier": getattr(trainer, "zo_dpga_noise_multiplier", None),
                "gaussian_std": getattr(trainer, "zo_dpga_gaussian_std", None),
                "accounting_convention": "opacus_poisson_accountant_with_fixed_without_replacement_batches",
            }
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.model.parameters())
        record = {
            "format": "zo_dpga-llama-benchmark-v1",
            "experiment_suite": self.args.experiment_suite,
            "experiment_id": self.args.experiment_id,
            "model_name": self.args.model_name,
            "task": self.args.task_name,
            "method": method,
            "mode": mode,
            "seed": self.args.seed,
            "train_set_seed": self.args.train_set_seed,
            "model_load_seconds": self.model_load_seconds,
            "training_wall_seconds": wall_seconds,
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "trainer_metrics": train_output.metrics,
            "parameters": {
                "total": total,
                "trainable": trainable,
                "trainable_fraction": trainable / total,
            },
            "hyperparameters": {
                "num_train": self.args.num_train,
                "num_dev": self.args.num_dev,
                "num_eval": self.args.num_eval,
                "max_length": self.args.max_length,
                "max_steps": self.args.max_steps,
                "batch_size_per_device": self.args.per_device_train_batch_size,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "learning_rate": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "zo_eps": self.args.zo_eps,
                "lora_r": self.args.lora_r if self.args.lora else None,
                "lora_alpha": self.args.lora_alpha if self.args.lora else None,
                "lora_target_modules": self.args.lora_target_modules if self.args.lora else None,
                "lora_num_layers": self.args.lora_num_layers if self.args.lora else None,
                "num_prefix": self.args.num_prefix if self.args.prefix_tuning else None,
            },
            "privacy": privacy,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": hf_transformers.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                "gpu_count": torch.cuda.device_count(),
                "gpus": gpu_stats,
            },
        }
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(os.path.join(self.args.output_dir, "benchmark.json"), "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        logger.info("Wrote benchmark record to %s", os.path.join(self.args.output_dir, "benchmark.json"))


    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            config = AutoConfig.from_pretrained(self.args.model_name)
            if config.model_type != "llama":
                raise ValueError(f"This entry point only supports Llama models, got model_type={config.model_type!r}")
            enabled_peft_modes = sum((self.args.prefix_tuning, self.args.head_tuning, self.args.lora))
            if enabled_peft_modes > 1:
                raise ValueError("Choose only one parameter-efficient mode: prefix, head tuning, or LoRA")
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.load_int8 and not self.args.lora:
                raise ValueError("8-bit Llama loading is supported only with --lora")
            if self.args.no_auto_device:
                # No auto device (use for FSDP)
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    trust_remote_code=self.args.trust_remote_code,
                )
            else:
                # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                load_kwargs = dict(
                    config=config,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.args.trust_remote_code,
                )
                if torch.cuda.is_available():
                    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
                    load_kwargs.update(
                        device_map="auto",
                        max_memory={i: f"{max(free_in_GB - 5, 1)}GB" for i in range(torch.cuda.device_count())},
                    )
                if self.args.load_int8:
                    load_kwargs["load_in_8bit"] = True
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    **load_kwargs,
                )

            model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            use_fast=False,
            trust_remote_code=self.args.trust_remote_code,
        )
        # Reading tokenizer.pad_token while it is unset emits a warning in the
        # legacy Llama tokenizer, so inspect the backing special-token fields.
        if getattr(tokenizer, "_pad_token", None) is None:
            if getattr(tokenizer, "_unk_token", None) is None:
                raise ValueError("Llama tokenizer has neither a pad token nor an unk token")
            tokenizer.pad_token = tokenizer._unk_token
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            from src.prefix import PrefixTuning
            PrefixTuning(
                model,
                num_prefix=self.args.num_prefix,
                reparam=not self.args.no_reparam,
                init_by_real_act=self.args.prefix_init_by_real_act,
            )
        if self.args.lora:
            from src.lora import LoRA
            LoRA(
                model,
                r=self.args.lora_r,
                alpha=self.args.lora_alpha,
                float16=self.args.load_float16,
                target_modules=self.args.lora_target_modules,
                num_layers=self.args.lora_num_layers,
            )

        if self.args.head_tuning:
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is None:
                raise ValueError("Llama model does not expose output embeddings for head tuning")
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in output_embeddings.parameters():
                parameter.requires_grad = True
            logger.info("Head tuning enabled: only the Llama LM head is trainable")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable:,}/{total:,} ({100 * trainable / total:.4f}%)")

        return model, tokenizer


    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)
        

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]


    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """

        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")

        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        evaluation_start = time.perf_counter()
        # Prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        evaluation_seconds = time.perf_counter() - evaluation_start
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
            path = os.path.join(self.args.output_dir, "evaluation_benchmark.json")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    runs = json.load(handle)
            else:
                runs = []
            runs.append({
                "num_demonstrations": len(train_samples),
                "num_examples": len(eval_samples),
                "wall_seconds": evaluation_seconds,
                "examples_per_second": len(eval_samples) / evaluation_seconds if evaluation_seconds else None,
                "metrics": metrics,
            })
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(runs, handle, indent=2, cls=EnhancedJSONEncoder)
        return metrics


    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]


        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
        
        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = partial(forward_wrap_with_option_len_zo_dpga.__get__(self.model, type(self.model)), zo_dpga=self.args.zo_dpga)


        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        trainer = OurTrainer(
            model=self.model, 
            args=self.args,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
        )
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device_id)
                torch.cuda.synchronize(device_id)
        train_start = time.perf_counter()
        train_output = trainer.train(resume_from_checkpoint=last_checkpoint)
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device_id)
        training_wall_seconds = time.perf_counter() - train_start
        self._write_benchmark(trainer, train_output, training_wall_seconds, last_checkpoint)

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()
        
        # FSDP compatibility
        self.model = trainer.model 
        
        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward


def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag


def main():
    args = parse_args()

    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)

    # Initialize trainer and load model
    framework = Framework(args, task)

    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                if args.num_dev is not None:
                    # Dev samples
                    dev_samples = train_samples[-args.num_dev:] 
                    train_samples = train_samples[:-args.num_dev]
                else:
                    dev_samples = None

                # Training
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)

                if not args.no_eval:
                    if args.dev_only:
                        metrics = {
                            "dev_" + key: value
                            for key, value in framework.evaluate([], dev_samples).items()
                        }
                    else:
                        metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
                    if dev_samples is not None and not args.dev_only:
                        dev_metrics = framework.evaluate([], dev_samples) 
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
            else:
                if args.dev_only:
                    dev_samples = train_samples[-args.num_dev:]
                    metrics = {
                        "dev_" + key: value
                        for key, value in framework.evaluate([], dev_samples).items()
                    }
                else:
                    assert args.num_dev is None
                    # Zero-shot / in-context learning
                    metrics = framework.evaluate(train_samples, eval_samples)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" +  result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)

if __name__ == "__main__": 
    main()
