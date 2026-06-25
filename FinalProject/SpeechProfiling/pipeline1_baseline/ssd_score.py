"""Heuristic SSD probability scoring for Pipeline 1.

Inputs: ComparisonResult (from comparison.py) + acoustic features dict
        (from acoustic.py) + duration & syllable count (for rate)
        + optional align_features dict (from align.alignment_features).

Outputs: dict mapping each of 6 categories
  {Normal, Articulation, Phonological, CAS, Dysarthria, Fluency}
to a probability that sums to 1.0 (i.e. % out of 100). Plus a per-category
breakdown of the contributing features and weights, for explainability.

This is a RULE-BASED baseline. It encodes published clinical heuristics
adapted to Kannada-relevant phoneme classes. It is NOT a learned classifier.
The intent is to make the API stable so a learned classifier can replace
the rules later when labelled SSD data becomes available.

NORMAL_PRESETS: four calibrated threshold sets derived from Kannada healthy
speech (MILE + SPRING + mixed, n≈100 each) on 2026-05-03. Pass a preset
name to `score(..., threshold_set="mile_screening")`. Default preset is
`mile_screening` — chosen because MILE is clean read-speech with the
narrowest healthy-band, giving the most defensible "normal" baseline.

Calibration source: runs/calibration_thresholds.json
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp
from typing import Any, Dict, List, Optional

from .comparison import ComparisonResult, PatternCounts

CATEGORIES = [
    "Normal", "Articulation", "Phonological", "CAS", "Dysarthria", "Fluency",
]


# ── Calibrated threshold presets ────────────────────────────────────────
# Generated 2026-05-03 from MILE (n=98) / SPRING (n=95) / MIXED (n=97).
# CER/WER thresholds intentionally NOT used for Normal-signal weighting
# (corpus-baseline WER for SPRING is ~50% on healthy speech; ASR can't
# distinguish disfluency from disorder). They remain in the preset for
# downstream consumers but the scorer ignores them.

NORMAL_PRESETS: Dict[str, Dict[str, float]] = {
    # Original hand-tuned defaults (kept for reproducibility / regression).
    "default": {
        "char_error_rate_max": 0.05,
        "word_error_rate_max": 0.10,
        "pause_ratio_max": 0.35,
        "f0_std_min": 12.0,
        "speech_rate_min_sps": 2.5,
        "speech_rate_max_sps": 6.5,
        "jitter_max": 0.020,
        "shimmer_max": 0.080,
        "hnr_min_db": 12.0,
        "longest_pause_max_s": 1.2,
    },
    "mile_screening": {
        "char_error_rate_max": 0.0603,
        "word_error_rate_max": 0.6339,
        "pause_ratio_max": 0.2894,
        "f0_std_min": 14.155,
        "speech_rate_min_sps": 4.1532,
        "speech_rate_max_sps": 7.237,
        "jitter_max": 0.0245,
        "shimmer_max": 0.1358,
        "hnr_min_db": 8.73,
        "longest_pause_max_s": 0.896,
        "align_char_dur_cv_max": 0.6467,
        "align_articulation_cps_min": 7.068,
        "align_articulation_cps_max": 13.799,
        "align_intra_pause_total_max_s": 4.382,
        "align_intra_pause_max_s_max": 0.9015,
        "align_char_dur_max_s_max": 0.366,
    },
    "mile_diagnosis": {
        "char_error_rate_max": 0.0471,
        "word_error_rate_max": 0.4359,
        "pause_ratio_max": 0.2636,
        "f0_std_min": 16.08,
        "speech_rate_min_sps": 4.5123,
        "speech_rate_max_sps": 6.796,
        "jitter_max": 0.0231,
        "shimmer_max": 0.1139,
        "hnr_min_db": 10.029,
        "longest_pause_max_s": 0.704,
        "align_char_dur_cv_max": 0.6172,
        "align_articulation_cps_min": 7.3677,
        "align_articulation_cps_max": 12.7025,
        "align_intra_pause_total_max_s": 3.534,
        "align_intra_pause_max_s_max": 0.775,
        "align_char_dur_max_s_max": 0.343,
    },
    # Calibrated 2026-05-03 from n=95 healthy SPRING-INX-Kannada utts (see
    # runs/calib_spring.csv). SPRING is spontaneous, code-switched Kannada
    # with much higher feature variance than read-speech MILE — the bands
    # below are correspondingly wider, and word_error_rate_max saturates at
    # 1.0 because the SPRING corpus baseline WER on healthy speech is ~50 %.
    # Use these presets when you expect spontaneous / code-switched speech
    # at inference time and want a Normal band that is not biased by clean
    # read-speech statistics.
    "spring_screening": {
        "char_error_rate_max": 0.75,
        "word_error_rate_max": 1.0,
        "pause_ratio_max": 0.4017,
        "f0_std_min": 0.49,
        "speech_rate_min_sps": 1.6614,
        "speech_rate_max_sps": 8.4689,
        "jitter_max": 0.0344,
        "shimmer_max": 0.1605,
        "hnr_min_db": 5.978,
        "longest_pause_max_s": 0.6912,
        "align_char_dur_cv_max": 1.2389,
        "align_articulation_cps_min": 6.0686,
        "align_articulation_cps_max": 23.9075,
        "align_intra_pause_total_max_s": 2.403,
        "align_intra_pause_max_s_max": 0.96,
        "align_char_dur_max_s_max": 0.632,
    },
    "spring_diagnosis": {
        "char_error_rate_max": 0.4917,
        "word_error_rate_max": 1.0,
        "pause_ratio_max": 0.3117,
        "f0_std_min": 6.84,
        "speech_rate_min_sps": 2.5418,
        "speech_rate_max_sps": 7.5052,
        "jitter_max": 0.0282,
        "shimmer_max": 0.1505,
        "hnr_min_db": 7.61,
        "longest_pause_max_s": 0.576,
        "align_char_dur_cv_max": 1.1244,
        "align_articulation_cps_min": 8.696,
        "align_articulation_cps_max": 18.8516,
        "align_intra_pause_total_max_s": 1.868,
        "align_intra_pause_max_s_max": 0.818,
        "align_char_dur_max_s_max": 0.532,
    },
    "mixed_screening": {
        "char_error_rate_max": 0.4341,
        "word_error_rate_max": 1.0,
        "pause_ratio_max": 0.323,
        "f0_std_min": 7.38,
        "speech_rate_min_sps": 2.5836,
        "speech_rate_max_sps": 8.4236,
        "jitter_max": 0.0267,
        "shimmer_max": 0.1579,
        "hnr_min_db": 7.084,
        "longest_pause_max_s": 0.8,
        "align_char_dur_cv_max": 1.0408,
        "align_articulation_cps_min": 5.985,
        "align_articulation_cps_max": 19.5378,
        "align_intra_pause_total_max_s": 4.25,
        "align_intra_pause_max_s_max": 0.904,
        "align_char_dur_max_s_max": 0.442,
    },
    "mixed_diagnosis": {
        "char_error_rate_max": 0.3333,
        "word_error_rate_max": 0.94,
        "pause_ratio_max": 0.2809,
        "f0_std_min": 12.34,
        "speech_rate_min_sps": 3.1638,
        "speech_rate_max_sps": 7.3296,
        "jitter_max": 0.025,
        "shimmer_max": 0.1442,
        "hnr_min_db": 8.752,
        "longest_pause_max_s": 0.64,
        "align_char_dur_cv_max": 0.8489,
        "align_articulation_cps_min": 7.1522,
        "align_articulation_cps_max": 18.266,
        "align_intra_pause_total_max_s": 3.008,
        "align_intra_pause_max_s_max": 0.81,
        "align_char_dur_max_s_max": 0.368,
    },
}

DEFAULT_PRESET = "mile_screening"

# Backwards-compat module-level alias.
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


def _bump(value: float, threshold: float, scale: float = 1.0) -> float:
    """Returns 0 below threshold, ramps up linearly above."""
    return max(0.0, (value - threshold) * scale)


def _bump_below(value: float, threshold: float, scale: float = 1.0) -> float:
    """Returns 0 above threshold, ramps up linearly below."""
    return max(0.0, (threshold - value) * scale)


def score(comparison: ComparisonResult,
          acoustic: Dict[str, float],
          n_ref_syllables: int,
          align_features: Optional[Dict[str, float]] = None,
          threshold_set: str = DEFAULT_PRESET) -> SSDResult:
    """Compute heuristic SSD probabilities.

    threshold_set: one of NORMAL_PRESETS keys. Default 'mile_screening'.
    align_features: optional dict from align.alignment_features(); if None
                    or empty, alignment-derived signals are skipped.
    """
    if threshold_set not in NORMAL_PRESETS:
        raise KeyError(f"unknown threshold_set {threshold_set!r}; "
                       f"available: {list(NORMAL_PRESETS)}")
    NORM = NORMAL_PRESETS[threshold_set]
    af = align_features or {}

    pat = comparison.pattern
    cer = comparison.char.error_rate
    wer = comparison.word.error_rate
    syl_er = comparison.syllable.error_rate
    n_subs = comparison.char.substitutions
    n_dels = comparison.char.deletions
    n_ref_chars = max(comparison.char.n_ref, 1)

    duration = acoustic.get("duration_s", 1e-6) or 1e-6
    voiced = acoustic.get("voiced_dur_s", duration)
    pause_ratio = acoustic.get("pause_ratio", 0.0)
    longest_pause = acoustic.get("longest_pause_s", 0.0)
    n_pauses = int(acoustic.get("n_pauses", 0))
    f0_std = acoustic.get("f0_std_hz", 0.0)
    jitter = acoustic.get("jitter_local", 0.0)
    shimmer = acoustic.get("shimmer_local", 0.0)
    hnr = acoustic.get("hnr_mean_db", 25.0)
    speech_rate = (n_ref_syllables / max(voiced, 1e-3))

    # Alignment features (only present for some presets)
    dur_cv = af.get("align_char_dur_cv", 0.0)
    dur_max = af.get("align_char_dur_max_s", 0.0)
    art_cps = af.get("align_articulation_rate_cps", 0.0)
    intra_total = af.get("align_intra_pause_total_s", 0.0)
    intra_max = af.get("align_intra_pause_max_s", 0.0)

    ev = {c: CategoryEvidence() for c in CATEGORIES}

    # ── Articulation: phoneme-level substitutions ──
    art_sub_rate = n_subs / n_ref_chars
    ev["Articulation"].add("char_substitution_rate", 4.0, art_sub_rate)
    ev["Articulation"].add("retroflex_to_dental", 1.5, pat.retroflex_to_dental / n_ref_chars)
    ev["Articulation"].add("deaspiration",       1.2, pat.deaspiration / n_ref_chars)
    ev["Articulation"].add("fricative_subst",    1.2, pat.fricative_substitution / n_ref_chars)
    ev["Articulation"].add("approximant_subst",  0.8, pat.approximant_substitution / n_ref_chars)

    # ── Phonological ──
    ev["Phonological"].add("final_cons_deletion", 2.5, pat.final_consonant_deletion / n_ref_chars)
    ev["Phonological"].add("geminate_simplified", 2.0, pat.geminate_simplified / n_ref_chars)
    ev["Phonological"].add("char_deletion_rate",  1.2, n_dels / n_ref_chars)
    ev["Phonological"].add("nasal_subst",         1.0, pat.nasal_substitution / n_ref_chars)

    # ── CAS: vowel distortion + rate variability + sylable errors ──
    ev["CAS"].add("vowel_length_errors", 2.0,
                  (pat.vowel_length_short_for_long + pat.vowel_length_long_for_short) / n_ref_chars)
    ev["CAS"].add("syllable_error_rate", 1.5, syl_er)
    if speech_rate < NORM["speech_rate_min_sps"]:
        ev["CAS"].add("slow_rate", 1.0, NORM["speech_rate_min_sps"] - speech_rate)
    ev["CAS"].add("longest_pause_anomaly", 0.6,
                  _bump(longest_pause, NORM["longest_pause_max_s"]))
    if "align_char_dur_cv_max" in NORM and dur_cv > 0:
        ev["CAS"].add("phone_duration_irregular", 2.0,
                      _bump(dur_cv, NORM["align_char_dur_cv_max"], scale=2.0))
    if "align_char_dur_max_s_max" in NORM and dur_max > 0:
        ev["CAS"].add("phone_lengthening", 1.5,
                      _bump(dur_max, NORM["align_char_dur_max_s_max"], scale=2.0))

    # ── Dysarthria: voice quality + monotony + slow rate ──
    ev["Dysarthria"].add("jitter",  3.0, _bump(jitter,  NORM["jitter_max"], scale=20.0))
    ev["Dysarthria"].add("shimmer", 3.0, _bump(shimmer, NORM["shimmer_max"], scale=8.0))
    ev["Dysarthria"].add("low_hnr", 2.0, _bump_below(hnr, NORM["hnr_min_db"], scale=0.1))
    if f0_std < NORM["f0_std_min"]:
        ev["Dysarthria"].add("monotone_f0", 1.5, (NORM["f0_std_min"] - f0_std) / 10.0)
    if speech_rate < NORM["speech_rate_min_sps"]:
        ev["Dysarthria"].add("slow_rate", 1.5, NORM["speech_rate_min_sps"] - speech_rate)
    ev["Dysarthria"].add("char_subst_imprecise", 1.0, art_sub_rate)
    if "align_articulation_cps_min" in NORM and art_cps > 0:
        ev["Dysarthria"].add("slow_articulation", 1.0,
                             _bump_below(art_cps, NORM["align_articulation_cps_min"], scale=0.5))
    if "align_char_dur_cv_max" in NORM and dur_cv > 0:
        ev["Dysarthria"].add("phone_duration_irregular", 1.0,
                             _bump(dur_cv, NORM["align_char_dur_cv_max"], scale=2.0))

    # ── Fluency: pause anomalies + repetitions ──
    ev["Fluency"].add("pause_ratio_high", 2.0,
                      _bump(pause_ratio, NORM["pause_ratio_max"], scale=4.0))
    ev["Fluency"].add("longest_pause", 1.5,
                      _bump(longest_pause, NORM["longest_pause_max_s"]))
    ev["Fluency"].add("excessive_pauses", 1.0,
                      max(0.0, n_pauses - 2) / max(duration, 1.0))
    ev["Fluency"].add("char_insertion_rate", 1.0,
                      comparison.char.insertions / n_ref_chars)
    if "align_intra_pause_total_max_s" in NORM and intra_total > 0:
        ev["Fluency"].add("intra_utt_pauses", 1.5,
                          _bump(intra_total, NORM["align_intra_pause_total_max_s"], scale=0.5))
    if "align_intra_pause_max_s_max" in NORM and intra_max > 0:
        ev["Fluency"].add("max_intra_pause", 1.0,
                          _bump(intra_max, NORM["align_intra_pause_max_s_max"], scale=2.0))

    # ── Normal: rewarded when metrics fall inside healthy bands ──
    # WER/CER intentionally excluded — corpus-baseline ASR errors are not a
    # reliable signal at the SPRING/MIXED end of the calibration data.
    normal_signals = 0.0
    if pause_ratio <= NORM["pause_ratio_max"]:    normal_signals += 1.0
    if f0_std >= NORM["f0_std_min"]:              normal_signals += 1.0
    if NORM["speech_rate_min_sps"] <= speech_rate <= NORM["speech_rate_max_sps"]:
        normal_signals += 1.0
    if jitter <= NORM["jitter_max"]:              normal_signals += 1.0
    if shimmer <= NORM["shimmer_max"]:            normal_signals += 1.0
    if hnr >= NORM["hnr_min_db"]:                 normal_signals += 1.0
    if longest_pause <= NORM["longest_pause_max_s"]: normal_signals += 1.0
    # Bonus for alignment-derived normality if features are present
    if "align_char_dur_cv_max" in NORM and dur_cv > 0:
        if dur_cv <= NORM["align_char_dur_cv_max"]:    normal_signals += 1.0
    if "align_articulation_cps_min" in NORM and art_cps > 0:
        if NORM["align_articulation_cps_min"] <= art_cps <= NORM["align_articulation_cps_max"]:
            normal_signals += 1.0
    if "align_intra_pause_total_max_s" in NORM and intra_total >= 0:
        if intra_total <= NORM["align_intra_pause_total_max_s"]: normal_signals += 1.0

    ev["Normal"].add("clean_signal_count", 1.0, normal_signals)

    # Convert raw scores → softmax probabilities
    raw = {c: ev[c].score_raw for c in CATEGORIES}
    temperature = 1.5
    import math
    exps = {c: math.exp(raw[c] / temperature) for c in CATEGORIES}
    Z = sum(exps.values()) or 1.0
    probs = {c: exps[c] / Z for c in CATEGORIES}

    binary = {
        "Normal_pct": probs["Normal"] * 100.0,
        "SSD_any_pct": (1.0 - probs["Normal"]) * 100.0,
    }

    return SSDResult(
        probabilities={c: probs[c] * 100.0 for c in CATEGORIES},
        binary_normal_vs_ssd=binary,
        raw_scores=raw,
        contributors={c: ev[c].contributors for c in CATEGORIES},
        threshold_set=threshold_set,
    )
