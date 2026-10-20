# Active Subj Data

Only the following directories are consumed by the unified pipeline:

- `subj_raw_prepared/subj/none_a3c561096c597e10`: clean Subj train/validation/test splits.
- `subj_embedding_protected`: ClipText+ protected train split with clean validation/test splits.
- `subj_memorization_canary`: an optional deterministic mechanism diagnostic. It is not used by
  natural-data training or attack commands and must not replace the original-data result.

The two train files both contain 7,200 rows. Their labels and row order are aligned; every protected training sentence differs from its corresponding clean sentence. The matching clean-data fingerprints in both manifests are the provenance check used for source-recovery attacks.

Generated example-steering audit populations and the exact sampled LoRA/MeZO/DPZero member populations are written below `outputs/models/*/audit_data`. Attacks use those saved populations rather than resampling the source dataset.

The canary set is only a mechanism stress test, not a natural-data leakage estimate. The primary
experiments train and attack the unchanged text in `subj_raw_prepared`.
