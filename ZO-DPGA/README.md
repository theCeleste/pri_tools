# ZODPGA for Llama 2 7B

This directory adapts the OPT implementation to `meta-llama/Llama-2-7b-hf`.
It covers zero-shot/ICL evaluation, regular fine-tuning, LoRA, head tuning,
linear probing, direct KV prefix tuning, MeZO, and ZODPGA.

## Setup

Request access to Llama 2 on Hugging Face, then authenticate:

```bash
huggingface-cli login
pip install -r requirements.txt
```

The first implementation remains pinned to `transformers==4.28.1`, matching the
Trainer internals vendored by the original project. Upgrading Transformers requires
first replacing the copied training loop with a small `Trainer` extension.

## Recommended mode

For a 7B model, start with BF16 LoRA. Full-parameter zeroth-order training does
not retain backward activations, but it still perturbs and regenerates random
directions for every trainable parameter on every step.

```bash
cd llama
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=lora \
  BS=4 LR=1e-5 EPS=1e-3 bash examples/mezo.sh
```

ZODPGA:

```bash
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=lora \
  BS=4 LR=1e-5 EPS=1e-3 DP_EPS=6 DP_CLIP=10 \
  bash examples/zo_dpga.sh
```

Regular LoRA fine-tuning:

```bash
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=lora \
  BS=4 LR=1e-4 bash examples/finetune.sh
```

Head tuning uses the same scripts with `MODE=head`. Linear probing is available
through the direct `run.py` interface with `--linear_probing`.

Prefix tuning uses `MODE=prefix`. It provides trainable per-layer KV cache entries
in Llama's native key/value-head layout. The current implementation supports the
legacy tuple cache used by the pinned Transformers 4.28.1 environment and direct
KV parameters (`--no_reparam`). With `--prefix_init_by_real_act`, it runs a short
no-gradient pass over randomly sampled real token IDs and initializes every layer
from the resulting RoPE-transformed native Llama KV cache.

Important constraints:

- ZODPGA requires `--trainer zo`, `--only_train_option`, and gradient accumulation 1.
- ZODPGA automatically enables `dataloader_drop_last` so every privacy step uses
  the configured fixed batch size.
- `--load_float16` and `--load_bfloat16` are mutually exclusive; BF16 is recommended.
- 8-bit loading is accepted only with LoRA.
- The privacy log reports sample rate, noise multiplier, Gaussian standard deviation,
  epsilon, and delta. Candidate-expanded classification batches are still accounted
  in terms of original examples by the configured per-device batch size.

Every ZODPGA checkpoint includes `zo_dpga_privacy.json` with the sampling rate,
noise multiplier, Gaussian standard deviation, batch and dataset sizes, clipping
threshold, finite-difference epsilon, world size, and planned step count. Resume
requires these privacy-relevant arguments to match exactly. To extend a completed
privacy schedule, start a new run and explicitly compose the two privacy budgets;
do not treat it as an ordinary same-budget resume.

The task names and prompt templates remain the same as in `opt/`.

## Loader messages

With PyTorch 2.4, Transformers 4.28 may emit a `torch.load(...,
weights_only=False)` FutureWarning while loading legacy `.bin` checkpoint shards.
This call is inside Transformers 4.28 and cannot be changed through the model API.
Use only the official gated Llama checkpoint or another checkpoint you trust. Do
not globally suppress the warning for arbitrary model paths. A future Trainer/API
upgrade should prefer safetensors and a Transformers version that passes
`weights_only=True` where applicable.

SentencePiece may print `precompiled_charsmap is empty; use identity normalization`.
That is an informational message for the tokenizer bundled with the checkpoint,
not a training or tokenization failure.

The Llama entry point defaults to `--optim adamw_torch` instead of the deprecated
Transformers 4.28 AdamW implementation. It can still be overridden explicitly on
the command line.

## Adapter checkpoints

LoRA, LM-head, and Prefix runs save adapter-only checkpoints by default. Each
checkpoint contains `pytorch_model.bin`, `adapter_manifest.json`, model config,
trainer state, optimizer, and scheduler state. On resume, reconstruct the same
mode through the original command; the trainer validates the manifest and loads
only the declared adapter tensors. Use `--save_adapter_only False` to request a
full model checkpoint. Full-parameter fine-tuning always saves the full model.

Adapter-only saving currently targets ordinary single-process/device-map runs.
FSDP and DeepSpeed retain their native full/sharded checkpoint behavior.

After the tiny checkpoint tests pass, run one real Llama 2 checkpoint smoke test:

```bash
MODE=prefix bash examples/validate_7b_adapter_checkpoint.sh
```

The script uses a new timestamped output directory, saves after one MeZO step,
reconstructs the base model and Prefix module, resumes to step two, verifies both
trainer states, and prints the adapter weight file sizes.

## Formal experiments and performance benchmarks

The locked baseline matrix is defined in `configs/formal_experiments.yaml`:

- `dev_zero_shot_control`: 1 control on the exact tuning dev subset;
- `mezo_zo_eps_sweep`: 6 dev-only LoRA/Prefix finite-difference jobs;
- `head_dp_batch_sweep`: 2 dev-only Head batch-size jobs;
- `mezo_lr_sweep`: 6 dev-only LoRA/Prefix LR calibration jobs;
- `head_dp_clip_sweep`: 3 dev-only Head clipping calibration jobs;
- `zero_shot_control`: 1 evaluation-only control;
- `mezo_utility_pilot`: 3 seed-0, 1000-step utility pilots;
- `zo_dpga_utility_pilot`: 3 matched epsilon-6 utility pilots;
- `performance_smoke`: 3 short jobs, one for each PEFT mode;
- `mezo_baseline`: 9 jobs, three modes by three seeds;
- `zo_dpga_budget_sweep`: 36 jobs, three modes by four nominal epsilon levels by three seeds.

List a suite without executing anything:

```bash
python run_formal_experiment.py --suite performance_smoke
```

Execute exactly one selected job:

```bash
python run_formal_experiment.py --suite performance_smoke --index 0 --run
```

Bulk execution is intentionally disabled. Existing output directories are refused
unless `--resume` is supplied, and a resume command must match the saved
`run_config.json`. Each run records the exact command and configuration SHA-256.

Every training run writes `benchmark.json` with model load time, training wall
time, Trainer throughput, parameter counts, software/hardware versions, per-GPU
peak allocated/reserved memory, and DP settings. Peak memory is reset after model
loading, so it represents the live model plus training peak, not transient loader
allocations. Post-training evaluation is stored separately in `eval_metrics.json`.

Aggregate completed runs with:

```bash
python summarize_benchmarks.py \
  --root result/formal \
  --output result/formal/summary
```

This produces per-run `runs.csv` and grouped `groups.csv`/`groups.json` summaries.
The formal DP epsilon values retain the current Opacus-accounting convention
documented above; this configuration does not change the sampling implementation.

After utility pilots pass, run formal jobs sequentially by stage. Inspect first:

```bash
python run_formal_stage.py --stage mezo_baseline
```

Start only with an exact confirmation token:

```bash
python run_formal_stage.py \
  --stage mezo_baseline \
  --run \
  --confirm mezo_baseline
```

The stage runner skips outputs that already contain valid config, benchmark,
evaluation, and metric artifacts. An incomplete output is resumed; any process
failure, hard timeout, or artifact validation failure stops the stage immediately
without automatic retry. Progress is written under `result/formal/stages/`.

After reviewing the MeZO baseline, run only ZODPGA seed 0:

```bash
python run_formal_stage.py --stage zo_dpga_seed0 --run --confirm zo_dpga_seed0
```

Seed 1 and seed 2 are separate stages and should start only after seed-0 review.

## Staged LoRA + ZODPGA parameter matrix

The tuning plan in `参数矩阵.txt` is implemented by
`configs/lora_zo_dpga_matrix.yaml`.  All four stages use 1000 train examples,
the held-out 500-example dev subset, nominal epsilon 6, and never inspect the
formal evaluation set.

List or run stage 1:

```bash
python run_lora_zo_dpga_matrix.py --stage stage1
python run_lora_zo_dpga_matrix.py --stage stage1 --index 0 --run
```

To execute the whole stage sequentially with fail-fast artifact validation:

```bash
python run_lora_zo_dpga_stage.py --stage stage1 --run --confirm stage1
```

After a stage completes, rank its dev results and write the selected parameters
to `configs/lora_zo_dpga_selection.yaml`:

```bash
python summarize_lora_zo_dpga_matrix.py --stage stage1 --update-selection
python run_lora_zo_dpga_stage.py --stage stage2 --run --confirm stage2
python summarize_lora_zo_dpga_matrix.py --stage stage2 --update-selection
python run_lora_zo_dpga_stage.py --stage stage3 --run --confirm stage3
python summarize_lora_zo_dpga_matrix.py --stage stage3 --update-selection
python run_lora_zo_dpga_stage.py --stage final --run --confirm final
python summarize_lora_zo_dpga_matrix.py --stage final
```

Selection updates are explicit.  Review each ranked CSV/JSON summary before
starting the next stage.  The final summary records mean dev accuracy, seed
standard deviation, the number of seeds above 63.4%, and whether the configured
64.5%/two-seed success rule was met.

## Round-two LoRA + ZODPGA exploration

`configs/lora_zo_dpga_exploration.yaml` continues from the first-round winner
without overwriting its outputs. All stages retain the 1000-train/500-dev split
and nominal epsilon 6:

1. `boundary`: 12 LR/clipping/finite-difference boundary jobs;
2. `structure`: 6 q-only versus q+v and last-8/last-16/all-layer jobs;
3. `schedule`: 6 constant/linear/cosine and warmup combinations;
4. `confirm`: the best two configurations over seeds 0/1/2.

Run and select each stage in order:

```bash
python run_lora_zo_dpga_exploration.py --stage boundary
python run_lora_zo_dpga_exploration_stage.py --stage boundary --run --confirm boundary
python summarize_lora_zo_dpga_exploration.py --stage boundary --update-selection

python run_lora_zo_dpga_exploration_stage.py --stage structure --run --confirm structure
python summarize_lora_zo_dpga_exploration.py --stage structure --update-selection

python run_lora_zo_dpga_exploration_stage.py --stage schedule --run --confirm schedule
python summarize_lora_zo_dpga_exploration.py --stage schedule --update-selection

python run_lora_zo_dpga_exploration_stage.py --stage confirm --run --confirm confirm
python summarize_lora_zo_dpga_exploration.py --stage confirm
```

Review every ranked dev summary before updating the selection file. The confirm
summary reports whether a candidate beats the first-round three-seed mean of
80.07%. Formal test data remains excluded throughout exploration.

### LoRA + MeZO-only exploration

For non-private zeroth-order LoRA accuracy, use the dedicated workflow below.
Selection uses only the 500-example development subset and writes to
`result/lora_mezo_exploration/`:

```bash
python run_lora_mezo_exploration.py --stage optimizer
python run_lora_mezo_exploration_stage.py --stage optimizer --run --confirm optimizer
python summarize_lora_mezo_exploration.py --stage optimizer --update-selection
python run_lora_mezo_exploration_stage.py --stage structure --run --confirm structure
python summarize_lora_mezo_exploration.py --stage structure --update-selection
python run_lora_mezo_exploration_stage.py --stage budget --run --confirm budget
python summarize_lora_mezo_exploration.py --stage budget --update-selection
python run_lora_mezo_exploration_stage.py --stage confirm --run --confirm confirm
python summarize_lora_mezo_exploration.py --stage confirm
```

Every job is explicitly `--trainer zo --lora`; no ZODPGA clipping, noise, or
privacy-budget arguments are included. The stages search MeZO `learning_rate`
and `zo_eps`, LoRA rank/projections/layers, then batch size, horizon, and
scheduler before confirming two finalists over seeds 0/1/2.

### LoRA + MeZO refinement

After the first MeZO exploration completes, continue from its selected
`1e-4` learning rate, `3e-3` finite-difference scale, and rank-16 last-16-layer
adapter with the refinement workflow:

```bash
python run_lora_mezo_refinement_stage.py --stage boundary --run --confirm boundary
python summarize_lora_mezo_refinement.py --stage boundary --update-selection
python run_lora_mezo_refinement_stage.py --stage architecture --run --confirm architecture
python summarize_lora_mezo_refinement.py --stage architecture --update-selection
python run_lora_mezo_refinement_stage.py --stage scale --run --confirm scale
python summarize_lora_mezo_refinement.py --stage scale --update-selection
python run_lora_mezo_refinement_stage.py --stage horizon --run --confirm horizon
python summarize_lora_mezo_refinement.py --stage horizon --update-selection
python run_lora_mezo_refinement_stage.py --stage confirm --run --confirm confirm
python summarize_lora_mezo_refinement.py --stage confirm
```

This round extends the upper learning-rate/finite-difference boundaries, adds
K/O projection and layer-window variants, separates LoRA rank from alpha, and
tests 15,000/20,000-step horizons. It remains development-only and writes to
`result/lora_mezo_refinement/`.

### Frozen MeZO scale reproductions

The two seed-0 scale configurations that produced 89.2% and 90.0% development
accuracy are frozen together in `configs/lora_mezo_scale_reproductions.yaml`.
Inspect or execute either named variant independently:

```bash
python run_lora_mezo_scale_reproduction.py --variant r8_a16_892
python run_lora_mezo_scale_reproduction.py --variant r8_a16_892 --run

python run_lora_mezo_scale_reproduction.py --variant r16_a32_900
python run_lora_mezo_scale_reproduction.py --variant r16_a32_900 --run
```

Use `--output-dir` for an additional independent repeat. Existing output
directories are refused unless `--resume` is explicitly supplied. Each run
stores the expected reference metrics and configuration SHA-256 in
`run_config.json`.

For the 89.2% rank-8 configuration alone, an additional standalone snapshot is
available. It requires no variant selection:

```bash
python run_lora_mezo_892_reproduction.py
python run_lora_mezo_892_reproduction.py --run
```

Its default output is `result/reproduction/lora_mezo_892_seed0`. Use a fresh
`--output-dir` for each independent repeat.

### Third-round MeZO precision search

The confirmed refinement result is 90.47% mean dev accuracy over three seeds.
`configs/lora_mezo_precision.yaml` performs a local search around that result
and writes to the independent `result/lora_mezo_precision/` tree. Run each
stage only after reviewing and selecting the previous stage:

```bash
python run_lora_mezo_precision_stage.py --stage optimizer --run --confirm optimizer
python summarize_lora_mezo_precision.py --stage optimizer --update-selection

python run_lora_mezo_precision_stage.py --stage layers --run --confirm layers
python summarize_lora_mezo_precision.py --stage layers --update-selection

python run_lora_mezo_precision_stage.py --stage scale --run --confirm scale
python summarize_lora_mezo_precision.py --stage scale --update-selection

python run_lora_mezo_precision_stage.py --stage batch --run --confirm batch
python summarize_lora_mezo_precision.py --stage batch --update-selection

python run_lora_mezo_precision_stage.py --stage confirm --run --confirm confirm
python summarize_lora_mezo_precision.py --stage confirm
```

The stages refine LR/finite-difference scale, q/k/v layer windows, LoRA
rank/alpha, and batch sizes 8/12/16 before a two-candidate three-seed
confirmation. The final summary reports whether each candidate beats the
90.47% reference mean. This remains dev-only; repeated tuning makes formal-test
evaluation essential once the configuration is locked.

### ZODPGA-only precision search

For privacy-preserving exploration, use `configs/lora_zo_dpga_precision.yaml`.
Every job is ZODPGA with nominal epsilon 6, delta 1e-5, fixed-size batches,
gradient accumulation 1, and dev-only evaluation. MeZO results are not eligible
for stage selection. Run the gated stages in order:

```bash
python run_lora_zo_dpga_precision_stage.py --stage core --run --confirm core
python summarize_lora_zo_dpga_precision.py --stage core --update-selection

python run_lora_zo_dpga_precision_stage.py --stage privacy_update --run --confirm privacy_update
python summarize_lora_zo_dpga_precision.py --stage privacy_update --update-selection

python run_lora_zo_dpga_precision_stage.py --stage rank --run --confirm rank
python summarize_lora_zo_dpga_precision.py --stage rank --update-selection

python run_lora_zo_dpga_precision_stage.py --stage batch --run --confirm batch
python summarize_lora_zo_dpga_precision.py --stage batch --update-selection

python run_lora_zo_dpga_precision_stage.py --stage confirm --run --confirm confirm
python summarize_lora_zo_dpga_precision.py --stage confirm
```

The 12/9/3/4/6 jobs refine finite-difference scale, layer window, privacy
horizon, LR/clipping, LoRA rank, and DP batch size. Each batch candidate gets a
fresh privacy-noise calibration. The confirmation summary compares only with
the current ZODPGA three-seed mean of 82.8%.

### Frozen 85.8% reproduction

The exact seed-2 run that produced 85.8% dev accuracy is frozen in
`configs/lora_zo_dpga_858_reproduction.yaml`. Inspect or execute it with:

```bash
python run_lora_zo_dpga_858_reproduction.py
python run_lora_zo_dpga_858_reproduction.py --run
```

For another independent repeat, use a fresh directory instead of resuming:

```bash
python run_lora_zo_dpga_858_reproduction.py \
  --output-dir result/reproduction/lora_zo_dpga_858_seed2_repeat2 \
  --run
```
