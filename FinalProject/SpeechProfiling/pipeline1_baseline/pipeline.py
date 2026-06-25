"""Pipeline 1 orchestrator.

Single entry point: `run_pipeline(reference_text, audio_path)` returns a
structured `PipelineOutput` containing every intermediate artifact and the
two top-level tables (speech-properties + SSD-likelihood).

Stages run lazily: a heavy step (e.g. SSL embedding) is only invoked if
the caller asks for it via `include_ssl=True`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from FinalProject.shared.audio_io import TARGET_SR, duration_seconds, load_audio
from FinalProject.shared.text_norm import normalize

from .acoustic import all_acoustic_features
from .align import AlignedSegment, MFAAligner, alignment_features, ctc_align
from .asr import ASRModel, load_asr
from .comparison import ComparisonResult, compare
from .config import DEFAULT_ASR_KEY, all_models
from .g2p import text_to_chars, text_to_phonemes, text_to_syllables
from .ssd_score import SSDResult, score


# Cached ASR model (so calling run_pipeline multiple times in a session
# doesn't reload the 3.7 GB checkpoint each time). Pipeline 2 uses the
# same orchestrator with model_key pointing into USER_FINETUNED.
_asr_cache: Dict[str, ASRModel] = {}
_mfa_aligner: Optional[MFAAligner] = None


def _get_asr(model_key: str) -> ASRModel:
    if model_key in _asr_cache:
        return _asr_cache[model_key]
    registry = all_models()
    if model_key not in registry:
        raise KeyError(f"unknown model key {model_key!r}; available: {list(registry)}")
    asr = load_asr(registry[model_key])
    _asr_cache[model_key] = asr
    return asr


def _get_mfa_aligner() -> MFAAligner:
    global _mfa_aligner
    if _mfa_aligner is None:
        _mfa_aligner = MFAAligner()
    return _mfa_aligner


@dataclass
class PipelineOutput:
    reference_text: str
    hypothesis_text: str
    duration_s: float
    n_ref_words: int
    n_ref_syllables: int
    n_ref_chars: int
    n_ref_phonemes: int
    asr_model: str

    acoustic: Dict[str, float] = field(default_factory=dict)
    comparison: Optional[ComparisonResult] = None
    ssd: Optional[SSDResult] = None
    aligned: List[AlignedSegment] = field(default_factory=list)
    align_features: Dict[str, float] = field(default_factory=dict)
    align_backend: str = "ctc"
    ssl: Optional[Dict[str, Any]] = None  # populated when include_ssl=True


def run_pipeline(reference_text: str,
                 audio_path,
                 model_key: str = DEFAULT_ASR_KEY,
                 include_ssl: bool = False,
                 align_backend: str = "ctc",
                 threshold_set: str = "mile_screening") -> PipelineOutput:
    """Full Pipeline 1: text + audio → speech properties + SSD likelihood.

    align_backend:  'ctc' (fast, default) or 'mfa' (uses kannada_v2b.zip).
    threshold_set:  one of ssd_score.NORMAL_PRESETS keys
                    (mile_screening / mile_diagnosis /
                     mixed_screening / mixed_diagnosis / default).
    """
    audio_path = Path(audio_path)
    waveform, _ = load_audio(audio_path, target_sr=TARGET_SR)
    duration = duration_seconds(waveform, sr=TARGET_SR)

    asr = _get_asr(model_key)

    # 1) ASR
    hyp = asr.transcribe_tensor(waveform)

    # 2) Reference decompositions
    ref_norm = normalize(reference_text)
    ref_words = ref_norm.split()
    ref_sylls = text_to_syllables(ref_norm)
    ref_chars = text_to_chars(ref_norm)
    try:
        ref_phones = text_to_phonemes(ref_norm)
    except Exception:
        ref_phones = []

    # 3) Acoustic features
    acoustic = all_acoustic_features(waveform, sr=TARGET_SR)

    # 4) Reference-vs-hypothesis comparison
    comp = compare(reference_text, hyp)

    # 5) Forced alignment + derived per-phone timing features
    if align_backend == "mfa":
        aligned = _get_mfa_aligner().align(audio_path, reference_text)
    elif align_backend == "ctc":
        aligned = ctc_align(asr, audio_path, reference_text)
    else:
        raise ValueError(f"align_backend must be 'ctc' or 'mfa', got {align_backend!r}")
    align_feats = alignment_features(aligned)

    # 6) Optional SSL pooled embedding (heavy — load aqc model)
    ssl_out = None
    if include_ssl:
        from . import ssl_embed
        emb = ssl_embed.embed(waveform)
        ssl_out = {
            "name": "data2vec_aqc_kn_ssl",
            "norm": emb["norm"],
            "T": emb["T"],
            "D": emb["D"],
            # don't dump the full vector to JSON by default; expose stats only
            "mean_l2": float((emb["mean"] ** 2).sum() ** 0.5),
            "std_mean": float(emb["std"].mean()),
        }

    # 7) SSD heuristic scoring
    ssd = score(comp, acoustic, n_ref_syllables=len(ref_sylls),
                align_features=align_feats, threshold_set=threshold_set)

    return PipelineOutput(
        reference_text=ref_norm,
        hypothesis_text=hyp,
        duration_s=duration,
        n_ref_words=len(ref_words),
        n_ref_syllables=len(ref_sylls),
        n_ref_chars=len(ref_chars),
        n_ref_phonemes=len(ref_phones),
        asr_model=model_key,
        acoustic=acoustic,
        comparison=comp,
        ssd=ssd,
        aligned=aligned,
        align_features=align_feats,
        align_backend=align_backend,
        ssl=ssl_out,
    )


# ── Table formatting ────────────────────────────────────────────────────
def speech_properties_rows(out: PipelineOutput) -> List[Tuple[str, str, str]]:
    """Return rows as (group, property, value_str) for tabular display."""
    a = out.acoustic
    c = out.comparison
    af = out.align_features
    rate = out.n_ref_syllables / max(a.get("voiced_dur_s", 1e-3), 1e-3)
    rows = [
        ("Lexical", "Reference text", out.reference_text),
        ("Lexical", "Hypothesis (ASR)", out.hypothesis_text),
        ("Lexical", "Reference words", f"{out.n_ref_words}"),
        ("Lexical", "Reference syllables", f"{out.n_ref_syllables}"),
        ("Lexical", "Reference chars (Kannada-only)", f"{out.n_ref_chars}"),
        ("Lexical", "Reference phonemes (espeak)", f"{out.n_ref_phonemes}"),
        ("Errors",  "WER (word)",  f"{c.word.error_rate*100:.2f}%"),
        ("Errors",  "Syllable ER", f"{c.syllable.error_rate*100:.2f}%"),
        ("Errors",  "CER (char)",  f"{c.char.error_rate*100:.2f}%"),
        ("Errors",  "PER (phoneme, espeak)", f"{c.phoneme.error_rate*100:.2f}%"),
        ("Errors",  "Char ops S/D/I", f"{c.char.substitutions}/{c.char.deletions}/{c.char.insertions}"),
        ("Pattern", "Retroflex→dental subs", str(c.pattern.retroflex_to_dental)),
        ("Pattern", "Deaspirations",         str(c.pattern.deaspiration)),
        ("Pattern", "Fricative subs",        str(c.pattern.fricative_substitution)),
        ("Pattern", "Vowel-length errors",   str(c.pattern.vowel_length_short_for_long + c.pattern.vowel_length_long_for_short)),
        ("Pattern", "Geminate simplified",   str(c.pattern.geminate_simplified)),
        ("Pattern", "Final-cons deletions",  str(c.pattern.final_consonant_deletion)),
        ("Timing",  "Duration (s)",          f"{out.duration_s:.2f}"),
        ("Timing",  "Voiced duration (s)",   f"{a.get('voiced_dur_s',0):.2f}"),
        ("Timing",  "Pause ratio",           f"{a.get('pause_ratio',0)*100:.1f}%"),
        ("Timing",  "Longest pause (s)",     f"{a.get('longest_pause_s',0):.2f}"),
        ("Timing",  "Pauses (count)",        f"{int(a.get('n_pauses',0))}"),
        ("Timing",  "Speech rate (syll/s)",  f"{rate:.2f}"),
        ("Pitch",   "F0 mean (Hz)",          f"{a.get('f0_mean_hz',0):.1f}"),
        ("Pitch",   "F0 std (Hz)",           f"{a.get('f0_std_hz',0):.1f}"),
        ("Pitch",   "F0 range (Hz)",         f"{a.get('f0_range_hz',0):.1f}"),
        ("Voice",   "Jitter (local)",        f"{a.get('jitter_local',0):.4f}"),
        ("Voice",   "Shimmer (local)",       f"{a.get('shimmer_local',0):.4f}"),
        ("Voice",   "HNR (dB)",              f"{a.get('hnr_mean_db',0):.2f}"),
        ("Voice",   "F1/F2/F3 (Hz)",
            f"{a.get('f1_mean_hz',0):.0f}/{a.get('f2_mean_hz',0):.0f}/{a.get('f3_mean_hz',0):.0f}"),
        ("Spectral","RMS mean",              f"{a.get('rms_mean',0):.4f}"),
        ("Spectral","ZCR mean",              f"{a.get('zcr_mean',0):.4f}"),
        ("Spectral","Spectral centroid mean (Hz)", f"{a.get('spec_centroid_mean',0):.1f}"),
        (f"Align ({out.align_backend})", "Chars aligned / unaligned",
            f"{int(af.get('align_n_chars_aligned',0))}/{int(af.get('align_n_chars_unaligned',0))}"),
        (f"Align ({out.align_backend})", "Char duration mean ± std (s)",
            f"{af.get('align_char_dur_mean_s',0):.3f} ± {af.get('align_char_dur_std_s',0):.3f}"),
        (f"Align ({out.align_backend})", "Char duration CV",
            f"{af.get('align_char_dur_cv',0):.3f}"),
        (f"Align ({out.align_backend})", "Articulation rate (chars/s)",
            f"{af.get('align_articulation_rate_cps',0):.2f}"),
        (f"Align ({out.align_backend})", "Intra-utt pauses (count, total s)",
            f"{int(af.get('align_intra_pause_count',0))}, {af.get('align_intra_pause_total_s',0):.2f}"),
    ]
    return rows


def ssd_likelihood_rows(out: PipelineOutput) -> List[Tuple[str, str]]:
    s = out.ssd.probabilities
    rows = [(c, f"{s[c]:.2f}%") for c in
            ["Normal", "Articulation", "Phonological", "CAS", "Dysarthria", "Fluency"]]
    rows.append(("— total —", f"{sum(s.values()):.2f}%"))
    rows.append(("Binary: Normal", f"{out.ssd.binary_normal_vs_ssd['Normal_pct']:.2f}%"))
    rows.append(("Binary: SSD",    f"{out.ssd.binary_normal_vs_ssd['SSD_any_pct']:.2f}%"))
    return rows
