"""Command-line entry point for Pipeline 1.

Usage:
  /home/prouser1/miniconda3/envs/wav2/bin/python -m \
      FinalProject.SpeechProfiling.pipeline1_baseline.cli \
      --audio /path/to/audio.wav \
      --text "ಕನ್ನಡ ರೆಫರೆನ್ಸ್ ಸಾಲು" \
      [--include-ssl] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .pipeline import run_pipeline, speech_properties_rows, ssd_likelihood_rows


def _print_table(title: str, rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + list(rows)) for i in range(len(headers))]
    sep = "  " + "  ".join("-" * w for w in widths)
    print()
    print(f"  ┌─ {title} ─" + "─" * max(0, sum(widths) + 4 * len(headers) - len(title) - 4))
    print("  " + "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print(sep)
    for r in rows:
        print("  " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--model", default=None,
                    help="ASR model key (default: pipeline DEFAULT_ASR_KEY)")
    ap.add_argument("--include-ssl", action="store_true",
                    help="Load aqc SSL model and add embedding stats")
    ap.add_argument("--align-backend", choices=["ctc", "mfa"], default="ctc",
                    help="Forced-alignment backend: ctc (default, fast) or "
                         "mfa (uses kannada_v2b.zip, ~5s per utterance)")
    ap.add_argument("--threshold-set",
                    choices=["default", "mile_screening", "mile_diagnosis",
                             "mixed_screening", "mixed_diagnosis"],
                    default="mile_screening",
                    help="Calibrated NORMAL threshold set for SSD scoring")
    ap.add_argument("--json", type=Path, default=None,
                    help="If set, dump full output to this JSON file")
    args = ap.parse_args()

    if not args.audio.exists():
        sys.exit(f"audio not found: {args.audio}")

    kwargs = {"include_ssl": args.include_ssl,
              "align_backend": args.align_backend,
              "threshold_set": args.threshold_set}
    if args.model:
        kwargs["model_key"] = args.model

    out = run_pipeline(args.text, args.audio, **kwargs)

    sp_rows = speech_properties_rows(out)
    _print_table("Speech properties", sp_rows, ("Group", "Property", "Value"))

    ll_rows = ssd_likelihood_rows(out)
    _print_table("SSD likelihood", ll_rows, ("Category", "Probability"))

    print(f"\nASR model: {out.asr_model}")
    print(f"Hypothesis: {out.hypothesis_text}")

    if args.json:
        payload = {
            "reference_text": out.reference_text,
            "hypothesis_text": out.hypothesis_text,
            "duration_s": out.duration_s,
            "asr_model": out.asr_model,
            "n_ref_words": out.n_ref_words,
            "n_ref_syllables": out.n_ref_syllables,
            "n_ref_chars": out.n_ref_chars,
            "n_ref_phonemes": out.n_ref_phonemes,
            "acoustic": out.acoustic,
            "comparison": {
                "word":     asdict(out.comparison.word),
                "syllable": asdict(out.comparison.syllable),
                "char":     asdict(out.comparison.char),
                "phoneme":  asdict(out.comparison.phoneme),
                "pattern":  asdict(out.comparison.pattern),
            },
            "ssd": {
                "probabilities": out.ssd.probabilities,
                "binary":        out.ssd.binary_normal_vs_ssd,
                "raw_scores":    out.ssd.raw_scores,
                "contributors":  out.ssd.contributors,
            },
            "align_backend": out.align_backend,
            "align_features": out.align_features,
            "aligned": [
                {"char": s.char, "start_s": s.start_s, "end_s": s.end_s,
                 "matched": s.matched, "pred_char": s.pred_char}
                for s in out.aligned
            ],
            "ssl": out.ssl,
        }
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\n→ JSON saved: {args.json}")


if __name__ == "__main__":
    main()
