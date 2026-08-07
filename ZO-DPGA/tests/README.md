# Checkpoint validation

Run the offline tiny-Llama checkpoint suite from the `llama` directory:

```bash
python tests/validate_checkpoint.py --mode all
```

It validates LoRA, LM-head tuning, and direct-KV Prefix tuning without downloading
Llama 2 7B. For each mode it trains one MeZO step, saves `checkpoint-1`, rebuilds
the model and adapter, resumes to step two, verifies `checkpoint-2`, and checks
the exact family of trainable parameter names. It also exercises the
`load_best_model_at_end` adapter path so partial checkpoints are not interpreted
as incomplete full-model checkpoints.

Temporary checkpoints are written under `tests/_checkpoint_validation`. The test
deletes only the selected mode's directory under that test root before rerunning.

Once this tiny test passes, perform one real-model adapter checkpoint smoke test:

```bash
MODE=prefix bash examples/validate_7b_adapter_checkpoint.sh
```

Validate the zeroth-order and ZODPGA mathematical paths with:

```bash
python tests/validate_zo_math.py
```

This checks parameter restoration after symmetric perturbations, compares the
central finite difference against an autograd directional derivative, verifies
DP clipping, and confirms that ZODPGA receives one finite loss per example.

Validate the real Llama tokenizer boundary logic and causal-LM padding behavior:

```bash
python tests/validate_prompt_and_padding.py \
  --model_name meta-llama/Llama-2-7b-hf
```

This loads only the tokenizer from the named checkpoint. Model-side checks use a
tiny local Llama configuration. It verifies answer token boundaries, left
truncation, BOS preservation, padding-invariant option loss, genuine token ID 0,
and candidate-expanded classification loss cardinality.

Validate generation and cache behavior without downloading model weights:

```bash
python tests/validate_generation.py
```

This checks ordinary generation, the initial Prefix KV-cache length, three-token
Prefix cache continuation with a left-padded batch, and the project's batched
`zo_forward_nondiff` generation/F1 objective.

Run the complete parameter-efficient training matrix with:

```bash
python tests/validate_training_matrix.py
```

This runs one tiny-Llama step for all nine combinations of LoRA, LM-head tuning,
and Prefix tuning with regular backpropagation, MeZO, and ZODPGA. Every combination
must produce a finite loss, reach global step one, and modify at least one of its
declared trainable tensors.

Validate strict fixed-batch privacy and checkpoint metadata with:

```bash
python tests/validate_dp_privacy.py
```

This verifies `drop_last`, saved sampling/noise fields, exact-configuration
adapter restore, rejection of privacy-changing resume arguments, and recovery of
the original sample count from candidate-expanded batches.

Validate formal experiment matrix expansion without launching training:

```bash
python tests/validate_formal_config.py
```

This checks the expected 3/9/36 suite sizes, globally unique experiment/output
identities, required reproducibility flags, and separation of MeZO and ZODPGA commands.

Validate the staged LoRA + ZODPGA tuning matrix with:

```bash
python tests/validate_lora_zo_dpga_matrix.py
```

This checks the planned 16/12/3/6 job counts, command flags, fixed nominal
epsilon, unique identities, and the equal 40,000-example budgets in stage 3.

Validate the round-two exploration matrix with:

```bash
python tests/validate_lora_zo_dpga_exploration.py
```

This checks the 12/6/6/6 stage sizes, LoRA target projection/layer flags,
unique identities, dev-only evaluation, epsilon 6, and accumulation 1.

Validate the dedicated non-private LoRA + MeZO exploration matrix with:

```bash
python tests/validate_lora_mezo_exploration.py
```

This checks the 12/8/6/6 stage sizes, MeZO-only commands, LoRA structure flags,
dev-only evaluation, unique identities, and the absence of ZODPGA arguments.

Validate the follow-up LoRA + MeZO refinement matrix with:

```bash
python tests/validate_lora_mezo_refinement.py
```

This checks the 12/7/7/6/6 stage sizes, selection inheritance, MeZO-only
commands, development-only evaluation, and unique experiment identities.

Validate the third-round MeZO precision matrix with:

```bash
python tests/validate_lora_mezo_precision.py
```

This checks the 12/5/6/3/6 stage sizes, gated selection inheritance, MeZO-only
commands, q/k/v structure flags, dev-only evaluation, and unique identities.

Validate the ZODPGA-only precision matrix with:

```bash
python tests/validate_lora_zo_dpga_precision.py
```

This checks the 12/9/3/4/6 stage sizes, selection inheritance, mandatory
ZODPGA flags, fixed epsilon/delta, accumulation 1, dev-only evaluation, and
unique identities.

Validate the two frozen MeZO scale reproduction commands with:

```bash
python tests/validate_lora_mezo_scale_reproductions.py
```

This locks the shared optimizer/data settings, the rank-8/rank-16 adapter
differences, expected reference metrics, and absence of ZODPGA flags.

Validate the standalone 89.2% rank-8 reproduction with:

```bash
python tests/validate_lora_mezo_892_reproduction.py
```
