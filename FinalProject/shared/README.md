# shared/ — Cross-cutting Utilities

Code and assets that are imported across multiple papers. Anything that ends up here should serve **at least two** of the four paper tracks; otherwise it belongs in the track folder.

## Contents

| Path | Purpose |
|------|---------|
| `audio_io.py` | unified audio loading + `TARGET_SR` constant (16 kHz mono); used by every pipeline |
| `text_norm.py` | Kannada-specific text normalisation, `KANNADA_RANGE` constants |
| `evaluation.py` | shared metric helpers (WER / CER) so multiple tracks compute them identically |
| `fairseq_bootstrap.py` | adds the SPRING fairseq fork to `sys.path` so checkpoints load with the right user-dir |
| `mfa_kannada/` | Kannada MFA acoustic model + training pipeline (see below) |

## `mfa_kannada/`

```
mfa_kannada/
├── prepare.py              # build MFA corpus from manifest (used for v1)
└── runs/v1/
    ├── corpus/              # 724 spk × 20k utt MILE-derived corpus
    ├── pron_dict.txt        # char-as-phone Kannada dict (65 entries)
    ├── train_v2b.yaml       # MFA training config (beam=100, retry_beam=400)
    ├── train_v2b.sh         # idempotent launcher
    ├── train_v2b.log        # training output
    ├── kannada_v2b.zip      # ✅ trained acoustic model (61.6 MB)
    └── _tmp_v2b/            # MFA's working directory (safe to delete)
```

### Why beam=100 / retry_beam=400

For Indic / low-resource MFA training, the default beams (10 / 40) are too tight for the early-iteration monophone GMM. v2 ran with the defaults and 21 of 32 jobs got `-nan` log-likelihood at iter-1. v2b raised both beams 10× and trained cleanly in ≈3 hours. Recorded in memory so we don't repeat the cycle.

### Reproduction

```bash
/media/csedept/lab7/FinalProject/shared/mfa_kannada/runs/v1/train_v2b.sh
# logs to runs/v1/train_v2b.log; output to runs/v1/kannada_v2b.zip
# run takes ~3h on the workstation
```

### Consumers of the MFA model

| Consumer | Purpose |
|----------|---------|
| `pipeline1_baseline/align.py::MFAAligner` | per-utterance forced alignment for SSD scoring (Pipeline 1 + 2) |
| `MFASAT/MFASATnofinetune/` (planned) | phone-feature extraction over SPRING ASR |
| `MFASAT/` (planned) | phone-classification supervision signal during SSL fine-tune |
