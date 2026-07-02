---
title: PA-Profile — Phone-Aligned Speech Profiler
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# PA-Profile — Phone-Aligned Speech Profiler

> A live demo of a screener-agnostic speech-sound-disorder (SSD) pipeline
> validated on two languages: **Kannada** (low-resourced, healthy adult
> calibration with a rule-based scorer) and **English UltraSuite**
> (typically-developing children + clinical SSD children, with a learned
> binary classifier). The same Stages 1–7 backbone runs in both paths;
> only the language-specific layers and the Stage-8 head change.
>
> Originally submitted to **Interspeech 2026 — Show & Tell**.

---

## 🚀 Quick start

You have three ways to run PA-Profile. Pick the one that matches your situation.

| Option | When to use | Time |
|---|---|---|
| **1. Hugging Face Space** (web)  | You just want to click "Try it" — no install. | ~30 s |
| **2. Docker** (local)            | You want it offline on your own machine, one command. | ~15 min build, then instant |
| **3. Native Python + conda** (local) | You want to hack on the code and don't want Docker. | ~30 min setup |

### Option 1 — Hugging Face Space (browser)

Open **https://huggingface.co/spaces/onepunchgin/pa-profile-demo**, wait
for the Space to wake up (cold start: ~30 s on free CPU tier), then click
the **Sample utterances** tab and load any sample.

### Option 2 — Docker

```bash
git clone https://github.com/onepunchgin/pa-profile
cd pa-profile
docker compose up --build
# open http://localhost:7860 in your browser
```

The first build takes ~15 minutes (it installs MFA via conda). After that,
`docker compose up` starts the demo in seconds.

If you didn't commit the model weights to the image (`HF_HUB_PULL=1` in
`docker-compose.yml`), set `HF_TOKEN` in a `.env` file next to
`docker-compose.yml` so the build can pull the private Kannada checkpoint:

```bash
echo "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx" > .env
docker compose up --build
```

### Option 3 — Native Python + conda

```bash
git clone https://github.com/onepunchgin/pa-profile
cd pa-profile

# Conda env (because MFA needs Kaldi binaries, which only ship via conda-forge)
conda create -n pa-profile -c conda-forge -y \
    python=3.10 montreal-forced-aligner=2.2.17 postgresql
conda activate pa-profile

# Python deps
pip install -r requirements.txt

# Pull model weights (~4.5 GB)
export PA_PROFILE_HF_USER=onepunchgin
bash scripts/download_models.sh      # or: python scripts/download_models.py

# Pre-download the English MFA dictionary + acoustic model
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Launch
python app.py                        # then open http://localhost:7860
```

### Tour of the UI

The demo has three tabs:

- **Live demo** — upload audio (or record from mic), paste the reference
  text, hit **Run pipeline**. You get back a **Normal vs SSD-likelihood**
  table, **speech properties** (WER/CER, phonological pattern flags,
  alignment timing stats), the **per-phone alignment**, and the raw
  **scorer contributors**.
- **Sample utterances** — one-click loaders for 5 bundled samples
  (Kannada MILE × 2, Kannada single word, UXSSD child, UXTD child).
- **About** — architecture diagram + which models are wired into each
  language path.

---

## 📖 Deep dive

Everything below is for people who want to understand *what's actually
happening* inside the demo — researchers, devs, or curious users.

### What the system does

Given a recording of someone reading a known reference text, PA-Profile
runs an 8-stage pipeline:

```
                 ┌─────────────────────────────────────────┐
                 │  Stage 8: SSD-likelihood scorer (swap)  │
                 │  ─ Kannada: rule-based, calibrated       │
                 │  ─ English: learned MLP on UXSSD/UXTD    │
                 └────────────────────▲────────────────────┘
                                      │
       ┌──────────────────────────────┴────────────────────────────────┐
       │ Stages 1–7 — language-agnostic backbone                       │
       │ 1. Audio I/O (16 kHz mono)                                    │
       │ 2. ASR transcription (CTC)                                    │
       │ 3. Reference text normalisation + G2P                         │
       │ 4. Acoustic features (pitch, formants, intensity, voicing)    │
       │ 5. Forced alignment (MFA per-phone time boundaries)           │
       │ 6. Comparison ref↔hyp (WER, CER, phonological-pattern flags)  │
       │ 7. Alignment-derived timing stats (phone-dur CV, pauses, etc) │
       └───────────────────────────────────────────────────────────────┘
```

Stage 8 is **swappable** — its `score()` signature is frozen. To plug in
a new language or a new disorder, you swap:

- the language-specific input layers (ASR model + G2P + MFA model);
- the Stage-8 head (rule-based scorer ↔ learned classifier).

Stages 1–7 are the same code in every path.

### What's in the demo

| Path | ASR | Forced aligner | Stage 8 |
|---|---|---|---|
| **Kannada** | SPRING `data2vec_kn` (CTC) | Kannada MFA acoustic model (`kannada_v2b.zip`, trained from scratch for this work in 2 h 58 min, 61.6 MB) | Rule-based; calibrated on n=98 healthy MILE sentences |
| **English** | wav2vec2-base fine-tuned on **UXTD** typically-developing children | `english_us_arpa` MFA acoustic + dictionary | Learned MLP / LR — **AUC 0.78, precision 0.84** on speaker-disjoint held-out children (8 SSD + 58 TD train; 2 SSD + 10 TD test) |

### What's in this repo

```
pa-profile/                                       (= ShowAndTell2_portable/ on disk)
├── README.md                       ← this file
├── app.py                          ← Gradio entrypoint
├── requirements.txt                ← pinned Python deps
├── Dockerfile                      ← reproducible image (HF Spaces compatible)
├── docker-compose.yml              ← one-line local launch
├── FinalProject/                   ← pipeline code (importable as a package)
│   ├── shared/                     ← audio I/O, text normalisation, fairseq bootstrap
│   └── SpeechProfiling/
│       ├── pipeline1_baseline/     ← Kannada pipeline (Stages 1–8 rule-based)
│       └── Ultrasuite/
│           ├── pipeline1_baseline/ ← English pipeline (Stages 1–8 rule-based)
│           └── stage8_classifier/  ← Learned SSD-vs-TD head (drop-in for English)
├── models/                         ← model weights (~4.5 GB; LFS or downloaded)
│   ├── kannada_asr/                  SPRING Kannada data2vec .pt + dict + fairseq userdir
│   ├── kannada_mfa/                  Kannada MFA acoustic model + char-as-phone pron dict
│   ├── english_asr/                  UXTD-finetuned wav2vec2 (HF format)
│   └── stage8/                       MLP+LR joblib for SSD-vs-TD
├── samples/                        ← 5 sample WAVs + reference .txt
├── docs/                           ← architecture figure
└── scripts/
    ├── download_models.sh / .py    ← pull weights from HF Hub
    └── push_to_hf_hub.sh           ← (admin) one-time upload helper
```

### Model cards

The weights live in four Hugging Face Hub model repos:

| Repo | Size | License | What it is |
|---|---|---|---|
| `onepunchgin/pa-profile-kannada-asr` | 3.76 GB | SPRING-INX (consult upstream) | SPRING lab's data2vec Kannada ASR finetune. **Hosted privately** by default — needs an HF token to pull. |
| `onepunchgin/pa-profile-kannada-mfa` | 62 MB | Apache-2.0 (this work) | Kannada MFA acoustic model trained from scratch (`kannada_v2b.zip`) + char-as-phone pron dict |
| `onepunchgin/pa-profile-uxtd-wav2vec2` | 361 MB | Apache-2.0 (this work) | wav2vec2-base fine-tuned on UXTD typically-developing children |
| `onepunchgin/pa-profile-stage8-classifier` | <5 MB | Apache-2.0 (this work) | Binary SSD-vs-TD classifier (scaler + LR + MLP) on the 27-dim acoustic+alignment feature vector |

> **The SPRING Kannada checkpoint**: redistribution status is unclear and
> the default deployment hosts it privately. If you cannot obtain access,
> you can still run the English UltraSuite path standalone — set
> `system="English (UltraSuite)"` in the UI and never touch the Kannada
> radio button.

### Path-agnostic env vars

The pipelines resolve every model path through env vars with workstation
fallbacks. `app.py` seeds them from `./models/` at startup; Docker sets
them in the image; in a custom deployment you can override any of them:

| Var | Default (set by `app.py`) |
|---|---|
| `PA_PROFILE_ROOT` | folder containing `app.py` |
| `PA_PROFILE_KANNADA_ASR_CKPT` | `$ROOT/models/kannada_asr/SPRING_INX_data2vec_Kannada.pt` |
| `PA_PROFILE_KANNADA_DICT`     | `$ROOT/models/kannada_asr/SPRING_INX_Kannada_dict.txt` |
| `PA_PROFILE_DATA2VEC_USERDIR` | `$ROOT/models/kannada_asr/data2vec_userdir` |
| `PA_PROFILE_KANNADA_MFA_ZIP`  | `$ROOT/models/kannada_mfa/kannada_v2b.zip` |
| `PA_PROFILE_KANNADA_PRON_DICT`| `$ROOT/models/kannada_mfa/pron_dict.txt` |
| `PA_PROFILE_ENGLISH_ASR`      | `$ROOT/models/english_asr` |
| `PA_PROFILE_STAGE8_MODEL`     | `$ROOT/models/stage8/model.joblib` |
| `PA_PROFILE_MFA_BIN`          | empty → use `mfa` from `$PATH` |

### Hardware

| Tier | Latency per utterance | Notes |
|---|---|---|
| HF Spaces free CPU (2 vCPU / 16 GB) | ~10–30 s | Workable for the live demo. First request after a cold-start is slower. |
| HF Spaces paid GPU (T4, ~$0.40 /hr)  | ~1–3 s | Comfortable for back-to-back interactive use. |
| Local CPU laptop                     | ~20–60 s | Mostly limited by the data2vec encoder forward pass. |
| Local GPU                            | ~1 s   | The 3.76 GB checkpoint just fits in a 6 GB GPU. |

### Hosting your own HF Space

```bash
# 1. Create an empty Space (Docker SDK).
huggingface-cli repo create pa-profile-demo --type space --space_sdk docker

# 2. Add it as a git remote and push the repo.
git remote add space https://huggingface.co/spaces/onepunchgin/pa-profile-demo
git push space main

# 3. In Settings → Variables and secrets of the Space, add:
#       HF_TOKEN = hf_xxx  (only if the Kannada ASR repo is private)
```

The first Space build takes 20–40 min (conda + LFS pulls). Subsequent
builds are incremental.

---

## 🛠 Troubleshooting

### "MFA aligner not found"
You're in the native-install path and forgot to install MFA. Re-run the
conda step in *Quick start → Option 3*. MFA cannot be installed with
`pip alone`; it needs Kaldi binaries from `conda-forge`.

### "RuntimeError: fairseq … data2vec_audio architecture"
The fairseq architecture registration didn't run. Make sure
`PA_PROFILE_DATA2VEC_USERDIR` points at `models/kannada_asr/data2vec_userdir`
and that that directory contains `data2vec_ctc.py`.

### "CUDA out of memory"
The Kannada data2vec checkpoint is 3.76 GB. On <6 GB GPUs, force CPU:
```bash
CUDA_VISIBLE_DEVICES= python app.py
```

### "first request is slow"
Cold start downloads + loads the ~4 GB Kannada checkpoint into memory.
Subsequent requests are fast because the ASR model is cached in process.
On HF Spaces this happens once per cold-boot of the Space.

### "the Kannada example shows 'SSD-any 0%' even on healthy speech"
Either the audio is silent or MFA failed to align. Open the
**Per-phone alignment** accordion: if it's empty, MFA didn't finish.
Check that the reference text matches the audio language and that
`models/kannada_mfa/pron_dict.txt` covers every character of your text.

### HF Spaces build fails with "disk full"
The free-tier Space has 16 GB. The image with both models fits, but it's
tight. Either:
- Drop the Kannada path: comment out `PA_PROFILE_KANNADA_*` env vars in
  the Dockerfile and remove the Kannada radio choice in `app.py`.
- Move models out of the image: set `HF_HUB_PULL=1` in the Dockerfile
  build args and remove `models/` from the repo (HF Spaces will pull them
  at build time instead of bloating the image).

---

## 📚 Citation

If this work is useful for your research, please cite:

```bibtex
@inproceedings{pa-profile-2026,
  title  = {PA-Profile: A Phone-Aligned Speech Profiler with a Screener-
            Agnostic Stage-8 Head — Cross-Language Validation on Kannada
            and English UltraSuite},
  author = {Deepak Swaroop, Veena Thenkanidiyoor, Dileep A. D., H Muralikrishna},
  booktitle = {SPECOM},
  year   = {2026},
}
```

## 🤝 Acknowledgements

- **SPRING Lab, IIT Madras** — the Kannada data2vec ASR checkpoint
  (`SPRING_INX_data2vec_Kannada.pt`) and the SPRING-INX letter dictionary.
- **MILE Lab, IISc Bangalore** — the MILE Kannada test corpus used for
  rule-based scorer calibration.
- **UltraSuite (UXSSD + UXTD)** — Eshky et al., 2018; the canonical English
  child speech dataset used to train and validate the learned Stage-8 head.
- **Montreal Forced Aligner** — McAuliffe et al., 2017.
- **Gradio** — for the web UI.
- **HuggingFace** — for free model + Space hosting.

## License

Code: **Apache-2.0** (this repo, `LICENSE`).
Model weights: see individual model cards on Hugging Face Hub. The SPRING
Kannada ASR checkpoint is redistributed under SPRING-INX terms — consult
the upstream release before reusing.
