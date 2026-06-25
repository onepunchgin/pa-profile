"""Ultrasuite Pipeline 1 CLI.

Usage:
  /home/prouser1/miniconda3/envs/wav2/bin/python -m \
      FinalProject.SpeechProfiling.Ultrasuite.pipeline1_baseline.cli \
      --audio /home/prouser1/core-uxssd/core/01M/BL1/001A.wav \
      --text "elephant umbrella train swing" [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--text",  type=str,  required=True)
    ap.add_argument("--threshold-set", default="default_english_child")
    ap.add_argument("--use-learned", action="store_true",
                    help="Also evaluate the learned Stage-8 classifier (Phase 3 head).")
    ap.add_argument("--learned-model", choices=["mlp", "lr"], default="mlp")
    ap.add_argument("--hf-model", default=None,
                    help="Override ASR HF model path. Default = UXTD-finetuned (Phase 2).")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if not args.audio.exists():
        sys.exit(f"audio not found: {args.audio}")

    out = run_pipeline(args.text, args.audio,
                       hf_model=args.hf_model,
                       threshold_set=args.threshold_set,
                       use_learned=args.use_learned,
                       learned_model=args.learned_model)

    print(f"\nASR model: {out.asr_model}")
    print(f"Hypothesis: {out.hypothesis_text!r}")
    print(f"Reference:  {out.reference_text!r}")
    print(f"Duration: {out.duration_s:.2f}s   Words={out.n_ref_words}   "
          f"Syllables={out.n_ref_syllables}   Phonemes={out.n_ref_phonemes}")

    print(f"\n┌─ SSD likelihood ─")
    for cat, p in out.ssd.probabilities.items():
        print(f"  {cat:14s}  {p:6.2f}%")
    print(f"  {'— total —':14s}  {sum(out.ssd.probabilities.values()):6.2f}%")
    print(f"  {'Normal':14s}  {out.ssd.binary_normal_vs_ssd['Normal_pct']:6.2f}%")
    print(f"  {'SSD-any':14s}  {out.ssd.binary_normal_vs_ssd['SSD_any_pct']:6.2f}%")

    if out.learned is not None and "error" not in out.learned:
        print(f"\n┌─ Learned Stage-8 ({out.learned['model'].upper()}) ─")
        print(f"  P(SSD)       {out.learned['ssd_prob_pct']:6.2f}%")
        print(f"  P(TD/Normal) {out.learned['normal_prob_pct']:6.2f}%")
    elif out.learned and "error" in out.learned:
        print(f"\n[!] learned classifier error: {out.learned['error']}")

    print(f"\n┌─ Pattern counts (English child SSD) ─")
    for k, v in asdict(out.comparison.pattern).items():
        print(f"  {k:30s}  {v}")

    print(f"\n┌─ Alignment timing ({out.align_backend}) ─")
    af = out.align_features
    print(f"  chars aligned:    {int(af.get('align_n_chars_aligned',0))}")
    print(f"  dur mean ± std:    {af.get('align_char_dur_mean_s',0):.3f} ± "
          f"{af.get('align_char_dur_std_s',0):.3f} s")
    print(f"  dur CV:            {af.get('align_char_dur_cv',0):.3f}")
    print(f"  articulation cps:  {af.get('align_articulation_rate_cps',0):.2f}")
    print(f"  intra-utt pauses:  {int(af.get('align_intra_pause_count',0))} "
          f"(total {af.get('align_intra_pause_total_s',0):.2f}s)")

    if args.json:
        payload = {
            "reference_text": out.reference_text,
            "hypothesis_text": out.hypothesis_text,
            "duration_s": out.duration_s,
            "asr_model": out.asr_model,
            "align_backend": out.align_backend,
            "n_ref_words": out.n_ref_words,
            "n_ref_syllables": out.n_ref_syllables,
            "n_ref_phonemes": out.n_ref_phonemes,
            "acoustic": out.acoustic,
            "align_features": out.align_features,
            "ssd": {
                "probabilities": out.ssd.probabilities,
                "binary": out.ssd.binary_normal_vs_ssd,
                "raw_scores": out.ssd.raw_scores,
            },
            "aligned": [
                {"phone": s.char, "start_s": s.start_s, "end_s": s.end_s}
                for s in out.aligned
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\n→ JSON saved: {args.json}")


if __name__ == "__main__":
    main()
