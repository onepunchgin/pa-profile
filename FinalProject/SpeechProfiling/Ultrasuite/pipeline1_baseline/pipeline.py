"""Ultrasuite Pipeline 1 orchestrator (English child speech).

Single entry: `run_pipeline(text, audio)` → PipelineOutput.
Mirrors `FinalProject.SpeechProfiling.pipeline1_baseline.pipeline` exactly
so the Gradio demo and downstream tooling can swap targets by dotted
import path only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio

from FinalProject.SpeechProfiling.pipeline1_baseline.acoustic import (
    all_acoustic_features,
)
from FinalProject.SpeechProfiling.pipeline1_baseline.align import (
    AlignedSegment, alignment_features,
)

from .align import EnglishMFAAligner
from .asr import EnglishASRModel, load_english_asr, TARGET_SR
from .comparison import ComparisonResult, compare
from .g2p import n_syllables, text_to_phones, text_to_syllables, text_to_words
from .ssd_score import SSDResult, score

# Optional: learned Stage-8 head (Phase 3). Imported lazily so the
# rule-based path doesn't pay for sklearn + joblib.
_learned_clf = None


def _get_learned_classifier():
    global _learned_clf
    if _learned_clf is None:
        from FinalProject.SpeechProfiling.Ultrasuite.stage8_classifier.predict import (
            LearnedSSDClassifier,
        )
        _learned_clf = LearnedSSDClassifier()
    return _learned_clf

_aligner: Optional[EnglishMFAAligner] = None


def _get_mfa() -> EnglishMFAAligner:
    global _aligner
    if _aligner is None:
        _aligner = EnglishMFAAligner()
    return _aligner


@dataclass
class PipelineOutput:
    reference_text: str
    hypothesis_text: str
    duration_s: float
    n_ref_words: int
    n_ref_syllables: int
    n_ref_phonemes: int
    asr_model: str
    align_backend: str = "mfa"

    acoustic: Dict[str, float] = field(default_factory=dict)
    comparison: Optional[ComparisonResult] = None
    ssd: Optional[SSDResult] = None
    aligned: List[AlignedSegment] = field(default_factory=list)
    align_features: Dict[str, float] = field(default_factory=dict)
    learned: Optional[Dict[str, float]] = None  # populated when use_learned=True


def _load_audio_mono16k(audio_path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(audio_path))
    if wav.dim() > 1 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0)


def run_pipeline(reference_text: str,
                 audio_path,
                 hf_model: str = None,
                 align_backend: str = "mfa",
                 threshold_set: str = "default_english_child",
                 use_learned: bool = False,
                 learned_model: str = "mlp") -> PipelineOutput:
    """Full Ultrasuite Pipeline 1 for English child speech.

    use_learned: if True, also evaluate the learned Stage-8 classifier
                 trained on UXSSD vs UXTD speaker-disjoint data
                 (AUC 0.78 with `mlp`, 0.64 with `lr`). The rule-based
                 score is always returned; the learned probability is
                 returned alongside in PipelineOutput.learned.
    learned_model: 'mlp' (default) or 'lr'.
    """
    audio_path = Path(audio_path)
    waveform = _load_audio_mono16k(audio_path)
    duration = waveform.numel() / TARGET_SR

    asr = load_english_asr(hf_model) if hf_model else load_english_asr()
    hyp = asr.transcribe_tensor(waveform)

    ref_words = text_to_words(reference_text)
    ref_phones = text_to_phones(reference_text)
    n_ref_syll = n_syllables(reference_text)

    acoustic = all_acoustic_features(waveform, sr=TARGET_SR)
    comp = compare(reference_text, hyp)

    if align_backend == "mfa":
        aligned = _get_mfa().align(audio_path, reference_text)
    else:
        aligned = []   # CTC alignment not yet wired for English; MFA is default
    align_feats = alignment_features(aligned)

    ssd = score(comp, acoustic, n_ref_syllables=n_ref_syll,
                align_features=align_feats, threshold_set=threshold_set)

    learned = None
    if use_learned:
        try:
            learned = _get_learned_classifier().predict(
                acoustic=acoustic, align_features=align_feats, model=learned_model,
            )
        except Exception as e:
            learned = {"error": f"{type(e).__name__}: {e}"}

    return PipelineOutput(
        reference_text=reference_text,
        hypothesis_text=hyp,
        duration_s=duration,
        n_ref_words=len(ref_words),
        n_ref_syllables=n_ref_syll,
        n_ref_phonemes=len(ref_phones),
        asr_model=asr.name,
        align_backend=align_backend,
        acoustic=acoustic,
        comparison=comp,
        ssd=ssd,
        aligned=aligned,
        align_features=align_feats,
        learned=learned,
    )
