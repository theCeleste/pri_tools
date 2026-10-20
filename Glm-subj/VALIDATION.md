# Validation

## Confirmed structure

- The local base checkpoint identifies itself as `model_type=chatglm` and maps `AutoModelForCausalLM` to `ChatGLMForConditionalGeneration`.
- Clean and ClipText+ Subj splits have matching row counts and labels. Train has 7,200 aligned pairs; validation/test are clean and identical across the two directories.
- Active training uses causal next-token loss. No active file uses `AutoModelForSequenceClassification`, SST-2, Llama, OPT, `/home/wrc`, or `/home/sch`.
- Every training/build path saves the exact construction/member rows used by its matching attack.
- Active file and directory names contain ASCII only.

## Checks run without the Python environment

```bash
python3 -m py_compile privacy_attack/*.py attack_chatglm.py
bash -n run_privacy_audit.sh
python3 attack_chatglm.py --help
python3 -m privacy_attack.train_lora --dry-run --max-train-samples 8 --epochs 1
python3 -m privacy_attack.build_steering --dry-run --construction-samples 8
python3 -m privacy_attack.train_zo --dry-run --method dpzero \
  --output-dir /tmp/subj-dry-run --max-train-samples 8 --max-steps 2
```

## Runtime validation still required

The current machine environment has none of the packages in `requirements.txt`, so no real 6B checkpoint was loaded and no attack result has been fabricated. After installation, first run a small integration smoke test in a fresh output root:

```bash
MODEL_ROOT="$PWD/outputs/smoke/models" TRAIN_SAMPLES=8 EPOCHS=1 \
bash run_privacy_audit.sh train-raw

MODEL_ROOT="$PWD/outputs/smoke/models" ATTACK_ROOT="$PWD/outputs/smoke/attacks" \
ATTACK_TARGETS=2 SAMPLES_PER_PREFIX=1 MAX_NEW_TOKENS=32 \
bash run_privacy_audit.sh attack-raw
```

Then inspect `outputs/smoke/attacks/raw/prefix/summary.json` and `top_matches.txt` before committing GPU time to the complete matrix.
