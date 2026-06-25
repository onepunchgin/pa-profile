"""SSD scoring for English child speech (Ultrasuite).

Mirrors `pipeline1_baseline/ssd_score.py` shape but with English-child-
specific pattern weights. This is a Phase-1 *rule-based* scorer; Phase 3
of the Ultrasuite plan replaces it with a learned classifier trained on
UXSSD's actual SSD labels.

Threshold defaults are first-cut English-child values (no calibration
yet — calibrate with UXTD typically-developing data when downloaded).

The scorer signature mirrors the Kannada one EXACTLY so the same Stage-8
swap logic applies — feed `align_features` from the alignment_features()
helper unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .comparison import ComparisonResult

CATEGORIES = [
    "Normal", "Articulation", "Phonological", "CAS", "Dysarthria", "Fluency",
]

NORMAL_PRESETS: Dict[str, Dict[str, float]] = {
    # First-cut English child thresholds. Calibrate against UXTD when downloaded.
    "default_english_child": {
        "pause_ratio_max": 0.40,
        "f0_std_min": 25.0,            # children have higher pitch variance
        "speech_rate_min_sps": 2.5,
        "speech_rate_max_sps": 6.0,    # children read slower than adults
        "jitter_max": 0.030,           # children's jitter is naturally higher
        "shimmer_max": 0.150,
        "hnr_min_db": 10.0,
        "longest_pause_max_s": 1.5,
        "align_char_dur_cv_max": 0.80,
        "align_articulation_cps_min": 5.0,
        "align_articulation_cps_max": 18.0,
        "align_intra_pause_total_max_s": 5.0,
        "align_intra_pause_max_s_max": 1.0,
    },
}
DEFAULT_PRESET = "default_english_child"
NORMAL: Dict[str, float] = NORMAL_PRESETS[DEFAULT_PRESET]


@dataclass
class CategoryEvidence:
    score_raw: float = 0.0
    contributors: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, weight: float, value: float):
        self.score_raw += weight * value
        self.contributors.append({"feature": name, "weight": weight,
                                  "value": round(value, 4),
                                  "contribution": round(weight * value, 4)})


@dataclass
class SSDResult:
    probabilities: Dict[str, float]
    binary_normal_vs_ssd: Dict[str, float]
    raw_scores: Dict[str, float]
    contributors: Dict[str, List[Dict[str, Any]]]
    threshold_set: str = ""


def _bump(value, threshold, scale=1.0):
    return max(0.0, (value - threshold) * scale)


def _bump_below(value, threshold, scale=1.0):
    return max(0.0, (threshold - value) * scale)


def score(comparison: ComparisonResult,
          acoustic: Dict[str, float],
          n_ref_syllables: int,
          align_features: Optional[Dict[str, float]] = None,
          threshold_set: str = DEFAULT_PRESET) -> SSDResult:
    if threshold_set not in NORMAL_PRESETS:
        raise KeyError(f"unknown threshold_set {threshold_set!r}; "
                       f"available: {list(NORMAL_PRESETS)}")
    NORM = NORMAL_PRESETS[threshold_set]
    af = align_features or {}

    pat = comparison.pattern
    n_ref_phonemes = max(comparison.phoneme.n_ref, 1)
    syl_er = comparison.syllable.error_rate

    voiced = acoustic.get("voiced_dur_s", 1e-3) or 1e-3
    pause_ratio = acoustic.get("pause_ratio", 0.0)
    longest_pause = acoustic.get("longest_pause_s", 0.0)
    n_pauses = int(acoustic.get("n_pauses", 0))
    f0_std = acoustic.get("f0_std_hz", 0.0)
    jitter = acoustic.get("jitter_local", 0.0)
    shimmer = acoustic.get("shimmer_local", 0.0)
    hnr = acoustic.get("hnr_mean_db", 25.0)
    duration = acoustic.get("duration_s", 1e-6) or 1e-6
    speech_rate = n_ref_syllables / max(voiced, 1e-3)

    dur_cv = af.get("align_char_dur_cv", 0.0)
    art_cps = af.get("align_articulation_rate_cps", 0.0)
    intra_total = af.get("align_intra_pause_total_s", 0.0)
    intra_max = af.get("align_intra_pause_max_s", 0.0)

    ev = {c: CategoryEvidence() for c in CATEGORIES}

    # ── Articulation: phone-level substitutions (stopping, fronting, gliding) ──
    ev["Articulation"].add("stopping",       2.0, pat.stopping       / n_ref_phonemes)
    ev["Articulation"].add("fronting",       2.0, pat.fronting       / n_ref_phonemes)
    ev["Articulation"].add("gliding",        1.5, pat.gliding        / n_ref_phonemes)
    ev["Articulation"].add("voicing_change", 1.2, pat.voicing_change / n_ref_phonemes)

    # ── Phonological: structural patterns ──
    ev["Phonological"].add("cluster_reduction",        2.5, pat.cluster_reduction        / n_ref_phonemes)
    ev["Phonological"].add("final_consonant_deletion", 2.5, pat.final_consonant_deletion / n_ref_phonemes)
    ev["Phonological"].add("weak_syllable_deletion",   2.0, pat.weak_syllable_deletion   / max(n_ref_syllables, 1))
    ev["Phonological"].add("phoneme_deletion_rate",    1.0, comparison.phoneme.deletions / n_ref_phonemes)

    # ── CAS: vowel distortion + sylable errors + slow + irregular timing ──
    ev["CAS"].add("syllable_error_rate", 1.5, syl_er)
    if speech_rate < NORM["speech_rate_min_sps"]:
        ev["CAS"].add("slow_rate", 1.0, NORM["speech_rate_min_sps"] - speech_rate)
    if dur_cv > 0:
        ev["CAS"].add("phone_duration_irregular", 2.0,
                      _bump(dur_cv, NORM["align_char_dur_cv_max"], scale=2.0))

    # ── Dysarthria: voice quality + monotony + slow rate ──
    ev["Dysarthria"].add("jitter",  3.0, _bump(jitter,  NORM["jitter_max"], scale=20.0))
    ev["Dysarthria"].add("shimmer", 3.0, _bump(shimmer, NORM["shimmer_max"], scale=8.0))
    ev["Dysarthria"].add("low_hnr", 2.0, _bump_below(hnr, NORM["hnr_min_db"], scale=0.1))
    if f0_std < NORM["f0_std_min"]:
        ev["Dysarthria"].add("monotone_f0", 1.5, (NORM["f0_std_min"] - f0_std) / 10.0)
    if speech_rate < NORM["speech_rate_min_sps"]:
        ev["Dysarthria"].add("slow_rate", 1.5, NORM["speech_rate_min_sps"] - speech_rate)
    if art_cps > 0:
        ev["Dysarthria"].add("slow_articulation", 1.0,
                             _bump_below(art_cps, NORM["align_articulation_cps_min"], scale=0.5))

    # ── Fluency: pause anomalies ──
    ev["Fluency"].add("pause_ratio_high", 2.0,
                      _bump(pause_ratio, NORM["pause_ratio_max"], scale=4.0))
    ev["Fluency"].add("longest_pause", 1.5,
                      _bump(longest_pause, NORM["longest_pause_max_s"]))
    ev["Fluency"].add("excessive_pauses", 1.0,
                      max(0.0, n_pauses - 2) / max(duration, 1.0))
    if intra_total > 0:
        ev["Fluency"].add("intra_utt_pauses", 1.5,
                          _bump(intra_total, NORM["align_intra_pause_total_max_s"], scale=0.5))

    # ── Normal: rewards healthy values ──
    normal_signals = 0.0
    if pause_ratio   <= NORM["pause_ratio_max"]:    normal_signals += 1.0
    if f0_std        >= NORM["f0_std_min"]:         normal_signals += 1.0
    if NORM["speech_rate_min_sps"] <= speech_rate <= NORM["speech_rate_max_sps"]:
        normal_signals += 1.0
    if jitter        <= NORM["jitter_max"]:         normal_signals += 1.0
    if shimmer       <= NORM["shimmer_max"]:        normal_signals += 1.0
    if hnr           >= NORM["hnr_min_db"]:         normal_signals += 1.0
    if longest_pause <= NORM["longest_pause_max_s"]: normal_signals += 1.0
    if dur_cv > 0 and dur_cv <= NORM["align_char_dur_cv_max"]: normal_signals += 1.0
    if art_cps > 0 and NORM["align_articulation_cps_min"] <= art_cps <= NORM["align_articulation_cps_max"]:
        normal_signals += 1.0
    ev["Normal"].add("clean_signal_count", 1.0, normal_signals)

    raw = {c: ev[c].score_raw for c in CATEGORIES}
    temperature = 1.5
    exps = {c: math.exp(raw[c] / temperature) for c in CATEGORIES}
    Z = sum(exps.values()) or 1.0
    probs = {c: exps[c] / Z for c in CATEGORIES}
    binary = {
        "Normal_pct":  probs["Normal"] * 100.0,
        "SSD_any_pct": (1.0 - probs["Normal"]) * 100.0,
    }
    return SSDResult(
        probabilities={c: probs[c] * 100.0 for c in CATEGORIES},
        binary_normal_vs_ssd=binary,
        raw_scores=raw,
        contributors={c: ev[c].contributors for c in CATEGORIES},
        threshold_set=threshold_set,
    )
