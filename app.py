"""Portable Gradio demo for PA-Profile — Kannada + English UltraSuite.

Self-contained replica of ShowAndTell2/app.py:
  * model weights live under ./models/
  * sample audio under ./samples/
  * pipeline code under ./FinalProject/
  * all hardcoded /media/csedept/lab7 paths are reached through env vars
    (PA_PROFILE_*) that this script sets before importing the pipelines.

Run:
    python app.py               # localhost:7860
    python app.py --share       # public gradio.live URL
    python app.py --port 8000   # alternative port

Env overrides (Docker / HF Spaces set these; locally the defaults below win):
    PA_PROFILE_ROOT             root of this folder (default: parent of app.py)
    PA_PROFILE_KANNADA_ASR_CKPT  Kannada data2vec .pt
    PA_PROFILE_KANNADA_DICT      Kannada fairseq letter dict
    PA_PROFILE_DATA2VEC_USERDIR  fairseq user-dir for data2vec arch
    PA_PROFILE_KANNADA_MFA_ZIP   Kannada MFA acoustic zip
    PA_PROFILE_KANNADA_PRON_DICT Kannada char-as-phone pron dict
    PA_PROFILE_ENGLISH_ASR       UXTD-finetuned wav2vec2 directory
    PA_PROFILE_STAGE8_MODEL      stage-8 SSD-vs-TD joblib
    PA_PROFILE_MFA_BIN           dir containing `mfa`; empty → use PATH
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Tuple

# ── 1. Resolve portable root and seed env vars before any pipeline import ──
PORTABLE_ROOT = Path(os.environ.get(
    "PA_PROFILE_ROOT", str(Path(__file__).resolve().parent)
))
MODELS = PORTABLE_ROOT / "models"
SAMPLES = PORTABLE_ROOT / "samples"

# Each os.environ.setdefault leaves explicit overrides alone, so Docker /
# HF Spaces can still inject their own values via the Space settings.
os.environ.setdefault("PA_PROFILE_KANNADA_ASR_CKPT",
                      str(MODELS / "kannada_asr" / "SPRING_INX_data2vec_Kannada.pt"))
os.environ.setdefault("PA_PROFILE_KANNADA_DICT",
                      str(MODELS / "kannada_asr" / "SPRING_INX_Kannada_dict.txt"))
os.environ.setdefault("PA_PROFILE_DATA2VEC_USERDIR",
                      str(MODELS / "kannada_asr" / "data2vec_userdir"))
os.environ.setdefault("PA_PROFILE_KANNADA_MFA_ZIP",
                      str(MODELS / "kannada_mfa" / "kannada_v2b.zip"))
os.environ.setdefault("PA_PROFILE_KANNADA_PRON_DICT",
                      str(MODELS / "kannada_mfa" / "pron_dict.txt"))
os.environ.setdefault("PA_PROFILE_ENGLISH_ASR",
                      str(MODELS / "english_asr"))
os.environ.setdefault("PA_PROFILE_STAGE8_MODEL",
                      str(MODELS / "stage8" / "model.joblib"))
# Leave PA_PROFILE_MFA_BIN unset → MFAAligner will use `mfa` from PATH,
# which is what the Docker image installs into the base conda env.

# Make the bundled FinalProject package importable.
sys.path.insert(0, str(PORTABLE_ROOT))

import gradio as gr             # noqa: E402  (must follow env setup)
import pandas as pd             # noqa: E402

from FinalProject.SpeechProfiling.pipeline1_baseline.pipeline import (  # noqa: E402
    run_pipeline as run_pipeline_kannada,
    speech_properties_rows as kn_props,
)
from FinalProject.SpeechProfiling.pipeline1_baseline.config import (    # noqa: E402
    DEFAULT_ASR_KEY, all_models,
)
from FinalProject.SpeechProfiling.Ultrasuite.pipeline1_baseline.pipeline import (  # noqa: E402
    run_pipeline as run_pipeline_english,
)


# ── 2. Samples bundled in ./samples ──
SAMPLE_UTTS = [
    {
        "label": "Kannada / MILE healthy sentence #1",
        "system": "Kannada",
        "audio": str(SAMPLES / "mile_2786_0017.wav"),
        "text":  None,  # read from .txt sibling
    },
    {
        "label": "Kannada / MILE healthy sentence #2",
        "system": "Kannada",
        "audio": str(SAMPLES / "mile_2656_0088.wav"),
        "text":  None,
    },
    {
        "label": "Kannada / short word (user-elicited)",
        "system": "Kannada",
        "audio": str(SAMPLES / "kannada_user_word.wav"),
        "text":  None,
    },
    {
        "label": "English / UXSSD child #1 (clinical SSD)",
        "system": "English (UltraSuite)",
        "audio": str(SAMPLES / "uxssd_01M_BL1_001A.wav"),
        "text":  None,
    },
    {
        "label": "English / UXTD typically-developing child",
        "system": "English (UltraSuite)",
        "audio": str(SAMPLES / "uxtd_01M_001B.wav"),
        "text":  None,
    },
]


def _load_sample(idx: int) -> Tuple[str, str, str]:
    s = SAMPLE_UTTS[idx]
    text = s["text"]
    if text is None:
        wav = Path(s["audio"])
        txt = wav.with_suffix(".txt")
        if txt.exists():
            text = txt.read_text(encoding="utf-8").splitlines()[0].strip()
    return s["audio"], text or "", s["system"]


def _ssd_bars(probs: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "Category": list(probs.keys()),
        "Probability (%)": [round(v, 2) for v in probs.values()],
    })


def _alignment_table(out) -> pd.DataFrame:
    rows = []
    for s in out.aligned:
        rows.append({
            "char/phone": s.char,
            "start_s":   "—" if s.start_s is None else round(s.start_s, 3),
            "end_s":     "—" if s.end_s   is None else round(s.end_s, 3),
            "matched":   "Y" if s.matched else "N",
            "predicted": s.pred_char or "—",
        })
    return pd.DataFrame(rows)


def run_demo(system, audio_path, ref_text, kn_model, kn_threshold,
             en_use_learned, en_learned_model):
    if not audio_path:
        return gr.Markdown("Upload an audio file or pick a sample utterance."), None, None, None, None
    if not ref_text or not ref_text.strip():
        return gr.Markdown("Reference text required."), None, None, None, None
    try:
        if system == "Kannada":
            out = run_pipeline_kannada(
                ref_text, audio_path,
                model_key=kn_model,
                align_backend="mfa",
                threshold_set=kn_threshold,
            )
            sp_rows = kn_props(out)
            sp_df = pd.DataFrame(sp_rows, columns=["Group", "Property", "Value"])
            ssd_df = _ssd_bars(out.ssd.probabilities)
            binary = out.ssd.binary_normal_vs_ssd
            header = (
                f"### Kannada — Result: **{binary['Normal_pct']:.1f}%** Normal · "
                f"**{binary['SSD_any_pct']:.1f}%** SSD-any  \n"
                f"Threshold preset: `{kn_threshold}`  \n"
                f"`hypothesis: {out.hypothesis_text}`  \n"
                f"`reference:  {out.reference_text}`"
            )
            contributors = json.dumps(out.ssd.contributors, indent=2,
                                      ensure_ascii=False)
        else:  # English UltraSuite
            out = run_pipeline_english(
                ref_text, audio_path,
                use_learned=en_use_learned,
                learned_model=en_learned_model,
            )
            af = out.align_features
            sp_df = pd.DataFrame([
                ("Lexical",   "Reference",  out.reference_text),
                ("Lexical",   "Hypothesis", out.hypothesis_text),
                ("Errors",    "WER",        f"{out.comparison.word.error_rate*100:.1f}%"),
                ("Errors",    "CER",        f"{out.comparison.char.error_rate*100:.1f}%"),
                ("Pattern",   "Cluster reduction", str(out.comparison.pattern.cluster_reduction)),
                ("Pattern",   "Final-cons deletion", str(out.comparison.pattern.final_consonant_deletion)),
                ("Pattern",   "Stopping",   str(out.comparison.pattern.stopping)),
                ("Pattern",   "Fronting",   str(out.comparison.pattern.fronting)),
                ("Pattern",   "Gliding",    str(out.comparison.pattern.gliding)),
                ("Timing",    "Duration (s)", f"{out.duration_s:.2f}"),
                ("Align/MFA", "Phones aligned", f"{int(af.get('align_n_chars_aligned',0))}"),
                ("Align/MFA", "Phone dur CV",   f"{af.get('align_char_dur_cv',0):.3f}"),
                ("Align/MFA", "Articulation cps", f"{af.get('align_articulation_rate_cps',0):.2f}"),
                ("Align/MFA", "Intra-utt pauses",
                              f"{int(af.get('align_intra_pause_count',0))} (total {af.get('align_intra_pause_total_s',0):.2f}s)"),
            ], columns=["Group", "Property", "Value"])

            ssd_df = _ssd_bars(out.ssd.probabilities)
            rule_normal = out.ssd.binary_normal_vs_ssd['Normal_pct']
            if out.learned and "error" not in out.learned:
                ln = out.learned
                header = (
                    f"### English (UltraSuite) — Stage-8 head:  "
                    f"**rule {rule_normal:.1f}%** Normal · "
                    f"**learned {ln['ssd_prob_pct']:.1f}%** SSD-prob "
                    f"({ln['model'].upper()})  \n"
                    f"`hypothesis: {out.hypothesis_text}`  \n"
                    f"`reference:  {out.reference_text}`  \n"
                    f"_Learned classifier trained on UXSSD/UXTD speaker-disjoint; "
                    f"speaker-disjoint AUC 0.78._"
                )
            else:
                header = (
                    f"### English (UltraSuite) — Rule-based: **{rule_normal:.1f}%** Normal  \n"
                    f"`hypothesis: {out.hypothesis_text}`  \n"
                    f"`reference:  {out.reference_text}`"
                )
            contributors = json.dumps(out.ssd.contributors, indent=2,
                                      ensure_ascii=False)
        align_df = _alignment_table(out)
    except Exception as e:
        return gr.Markdown(f"Pipeline failed: `{type(e).__name__}: {e}`"), None, None, None, None

    return gr.Markdown(header), sp_df, ssd_df, align_df, contributors


def build_app():
    arch_png = PORTABLE_ROOT / "docs" / "pipeline1_stages.png"
    arch_block = (
        f"![architecture](file={arch_png})\n\n"
        if arch_png.exists() else ""
    )

    with gr.Blocks(title="PA-Profile — multi-language SSD demo") as app:
        gr.Markdown(
            "# PA-Profile — Phone-Aligned Speech Profiler\n\n"
            "Kannada (calibrated rule-based scorer) plus the English "
            "UltraSuite swap (learned Stage-8 binary SSD-vs-TD classifier, "
            "AUC 0.78 on speaker-disjoint held-out children). Same Stages "
            "1–7 backbone code in both; only the language-specific layers "
            "and the Stage-8 head change.  \n"
            "**Focus on:** low-resourced language + atypical (child SSD) speech.  \n"
            "**Conference theme:** *Speaking Together*"
        )

        with gr.Tab("Live demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    system = gr.Radio(
                        ["Kannada", "English (UltraSuite)"],
                        value="Kannada",
                        label="System",
                    )
                    audio = gr.Audio(label="Audio", type="filepath",
                                     sources=["upload", "microphone"])
                    text  = gr.Textbox(label="Reference text", lines=2,
                                       placeholder="Paste the reference text here…")
                    with gr.Accordion("Kannada-specific options", open=False):
                        kn_model = gr.Dropdown(
                            label="Kannada ASR model",
                            choices=list(all_models().keys()),
                            value=DEFAULT_ASR_KEY,
                        )
                        kn_threshold = gr.Dropdown(
                            label="Kannada threshold preset",
                            choices=["mile_screening", "mile_diagnosis",
                                     "spring_screening", "spring_diagnosis",
                                     "mixed_screening", "mixed_diagnosis",
                                     "default"],
                            value="mile_screening",
                        )
                    with gr.Accordion("English-specific options", open=False):
                        en_use_learned = gr.Checkbox(
                            label="Use learned Stage-8 (UltraSuite-trained classifier)",
                            value=True,
                        )
                        en_learned_model = gr.Radio(
                            ["mlp", "lr"], value="mlp",
                            label="Learned-classifier model",
                        )
                    run_btn = gr.Button("Run pipeline", variant="primary")
                with gr.Column(scale=2):
                    header_md = gr.Markdown()
                    with gr.Accordion("SSD likelihood / Stage-8 output", open=True):
                        ssd_table = gr.DataFrame(
                            headers=["Category", "Probability (%)"],
                            interactive=False)
                    with gr.Accordion("Speech properties", open=False):
                        sp_table  = gr.DataFrame(
                            headers=["Group", "Property", "Value"],
                            interactive=False)
                    with gr.Accordion("Per-phone alignment", open=False):
                        align_table = gr.DataFrame(
                            headers=["char/phone", "start_s", "end_s",
                                     "matched", "predicted"],
                            interactive=False)
                    with gr.Accordion("Scorer contributors (raw)", open=False):
                        contrib = gr.Code(language="json")

            run_btn.click(
                run_demo,
                inputs=[system, audio, text, kn_model, kn_threshold,
                        en_use_learned, en_learned_model],
                outputs=[header_md, sp_table, ssd_table, align_table, contrib],
            )

        with gr.Tab("Sample utterances"):
            gr.Markdown("Click any **Load** button to populate the Live demo "
                        "tab with that sample. The system radio is set for you.")
            for i, s in enumerate(SAMPLE_UTTS):
                with gr.Row():
                    gr.Markdown(f"**{s['label']}**  \n`{Path(s['audio']).name}`")
                    btn = gr.Button(f"Load #{i+1}")
                    def _make_loader(idx):
                        def _f():
                            a, t, sys_lbl = _load_sample(idx)
                            return a, t, sys_lbl
                        return _f
                    btn.click(_make_loader(i), outputs=[audio, text, system])

        with gr.Tab("About"):
            gr.Markdown(
                "## Architecture\n\n"
                f"{arch_block}"
                "Stages 1–7 are a reusable acoustic + alignment + comparison "
                "backbone. Stage 8 is a swappable scorer.\n\n"
                "## Demo systems\n\n"
                "- **Kannada (default):** SPRING `data2vec_kn` ASR + Kannada MFA "
                "(`kannada_v2b.zip`, trained for this work) + rule-based "
                "Stage 8 calibrated on n=98 healthy MILE sentences.\n"
                "- **English UltraSuite:** wav2vec2-base fine-tuned on UXTD "
                "typically-developing children + English MFA "
                "(`english_us_arpa`) + **learned** Stage-8 binary SSD-vs-TD "
                "MLP. Speaker-disjoint AUC 0.78 (8 SSD + 58 TD children, "
                "holdout 2 SSD + 10 TD speakers).\n\n"
                "## Why screener-agnostic\n\n"
                "Stage 8's `score()` API is frozen. The English path "
                "replaces only:\n"
                "- the language-specific input layers (ASR, G2P, MFA model);\n"
                "- the Stage-8 head (rule-based → learned MLP).\n\n"
                "Stages 1–7 (acoustic features, MFA forced alignment, "
                "comparison logic, alignment-derived timing features) are "
                "the same code in both paths.\n\n"
                "## Source\n\n"
                "Repository: https://github.com/onepunchgin/pa-profile  \n"
                "Model weights: huggingface.co/onepunchgin/pa-profile-*"
            )

    return app


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = ap.parse_args()
    build_app().launch(server_name="0.0.0.0", server_port=args.port,
                       share=args.share)
