# Pipeline 1 — Baseline SSD Screener

Single-utterance pipeline that takes (reference Kannada text, audio) and produces a **speech-properties table** plus a **6-class SSD-likelihood table** (probabilities sum to 100 %), with a binary `Normal vs SSD-any` collapse for screening.

This is the *baseline* — it uses **only** SPRING lab's pre-trained ASR (no user fine-tuning required) so it works the moment the workstation has fairseq + the SPRING checkpoint. Pipeline 2 mirrors this architecture but swaps in the user's finetuned models.

---

## Pipeline stages

```
text + audio
    │
    ├─► [1] ASR              data2vec_kn (SPRING) — hypothesis transcription
    │
    ├─► [2] Reference G2P    espeak phonemizer + Kannada syllabifier
    │                        produces: chars, syllables, phonemes
    │
    ├─► [3] Acoustic         librosa + Praat (parselmouth)
    │                        F0 / jitter / shimmer / HNR / pause stats / formants
    │
    ├─► [4] Forced align     CTC-greedy (default) OR MFA (kannada_v2b.zip)
    │                        per-char (start_s, end_s, matched, pred_char)
    │
    ├─► [5] Comparison       multi-granularity ref↔hyp diff:
    │                        word/syll/char/phoneme error rates
    │                        + Kannada-specific pattern counts
    │                          (retroflex↔dental, deaspiration, fricative
    │                           subs, vowel-length errors, gemination loss,
    │                           final-cons deletion)
    │
    ├─► [6] Align features   per-char dur mean / std / CV / max
    │   (from MFA output)    articulation rate (chars/sec)
    │                        intra-utt pause stats
    │
    ├─► [7] SSL embedding    OPTIONAL — data2vec_aqc, pooled
    │   (--include-ssl)      stats only by default (mean L2, std mean)
    │
    └─► [8] SSD scorer       rule-based; uses calibrated NORMAL_PRESETS
                             6-class softmax + binary collapse
```

A render lives at [`../../docs/pipeline1_stages.png`](../../docs/pipeline1_stages.png).

---

## Files

| File | Purpose |
|------|---------|
| `cli.py` | command-line entry: `python -m FinalProject.SpeechProfiling.pipeline1_baseline.cli ...` |
| `pipeline.py` | orchestrator — `run_pipeline(text, audio, ...) -> PipelineOutput` |
| `config.py` | model registry (incl. SPRING checkpoints + dict paths) |
| `asr.py` | fairseq adapter for SPRING data2vec / wav2vec2 / data2vec_aqc checkpoints |
| `g2p.py` | espeak phonemizer + Kannada syllable splitter |
| `acoustic.py` | librosa + parselmouth feature extraction |
| `align.py` | `ctc_align()`, `MFAAligner`, `alignment_features()` |
| `comparison.py` | multi-granularity diff + Kannada SSD-pattern counters |
| `ssd_score.py` | rule-based 6-class scorer + `NORMAL_PRESETS` (calibrated) |
| `ssl_embed.py` | data2vec_aqc pooled SSL embedding |
| `scripts/batch_eval.py` | batch over an `audio/` × `reference/` directory |
| `scripts/calibrate.py` | sample N utts from MILE/SPRING, dump features for threshold calibration |
| `scripts/analyze_calibration.py` | percentile-based threshold derivation from calibration CSVs |
| `runs/` | output CSVs/JSONs from above scripts |
| `_assets/` | static assets used by the pipeline |

---

## Key design choices

### Why rule-based, not learned?

The 6-class `ssd_score.score()` is intentionally rule-based with explainable per-feature contributors. The interface is a frozen `score(comparison, acoustic, n_ref_syllables, align_features=, threshold_set=) -> SSDResult`. A learned classifier can drop in behind that signature once labelled SSD data is available without touching anything upstream.

### Why ASR errors are *not* used as Normal-evidence

WER and CER are kept in the pipeline output for reporting, but they're **excluded from the Normal-signal counter** in `ssd_score.score()`. Calibration on SPRING_INX_R1 showed median WER ≈ 48 % on healthy speech — the ASR can't distinguish disfluency / code-switching from speech disorder, so weighting WER toward "abnormal" produces false positives across the board.

### Why MFA is the only valid alignment for duration features

CTC-greedy alignment emits one ~20 ms frame per token, so every char's duration is a constant. Any duration-CV / phone-lengthening feature derived from CTC is degenerate (uniform 0). The `MFAAligner` (subprocess to MFA 2.2.17 with `kannada_v2b.zip`) gives real per-phone intervals. CTC remains the default backend because it's faster (~0.5 s/utt vs ~20 s/utt) and fine for char-identity / position questions.

### NORMAL_PRESETS

Five presets in `ssd_score.NORMAL_PRESETS`:

| Preset | Source | Band | Use case |
|--------|--------|------|----------|
| `default` | hand-tuned originals | — | reproducibility / regression |
| `mile_screening` | MILE n=98 | 5 / 95 | **default** — balanced screening |
| `mile_diagnosis` | MILE n=98 | 10 / 90 | tighter, diagnostic depth |
| `mixed_screening` | MILE+SPRING n=97 | 5 / 95 | for noisier production audio |
| `mixed_diagnosis` | MILE+SPRING n=97 | 10 / 90 | tighter, mixed input |

Calibration script: `scripts/calibrate.py`. Analysis script: `scripts/analyze_calibration.py`. Both are idempotent (`calibrate.py` resumes from the last row in its CSV).

---

## Quick start

```bash
# Single utterance, MFA alignment, default thresholds, dump JSON
/home/prouser1/miniconda3/envs/wav2/bin/python -m \
  FinalProject.SpeechProfiling.pipeline1_baseline.cli \
  --audio /path/to/utt.wav \
  --text "ಕನ್ನಡ ಸಾಲು" \
  --align-backend mfa \
  --threshold-set mile_screening \
  --json out.json

# Batch over the SpeechProfiling/{audio,reference}/ directory
/home/prouser1/miniconda3/envs/wav2/bin/python -m \
  FinalProject.SpeechProfiling.pipeline1_baseline.scripts.batch_eval \
  --align-backend mfa --threshold-set mile_screening
```

---

## Outputs (sample)

The CLI prints two tables and optionally dumps JSON:

```
┌─ Speech properties ───────────────────────────
  Group         Property                Value
  ----------    ----------------------  ----------------
  Lexical       Reference text          ಕನ್ನಡ ಸಾಲು
  Lexical       Hypothesis (ASR)        ...
  Errors        WER (word)              16.67%
  Errors        CER (char)              4.55%
  Pattern       Retroflex→dental subs   0
  Pattern       Geminate simplified     1
  Timing        Duration (s)            2.34
  Timing        Pause ratio             12.0%
  Pitch         F0 mean (Hz)            142.3
  Voice         Jitter (local)          0.0181
  ...
  Align (mfa)   Char duration CV        0.523
  Align (mfa)   Articulation rate (cps) 9.84

┌─ SSD likelihood ──────────────────────────────
  Category          Probability
  ----------------  -----------
  Normal            74.21%
  Articulation       4.10%
  Phonological       2.92%
  CAS                3.07%
  Dysarthria        12.04%
  Fluency            3.67%
  — total —        100.01%
  Binary: Normal    74.21%
  Binary: SSD       25.79%
```
