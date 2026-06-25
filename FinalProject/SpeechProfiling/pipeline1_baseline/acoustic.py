"""Acoustic feature extraction for Pipeline 1.

Two extractors:
  - librosa-based: pitch (pyin), MFCC stats, RMS energy, spectral
    centroid/bandwidth, ZCR, speaking rate proxies, pause structure.
  - parselmouth-based (Praat): jitter, shimmer, HNR, formants — voice-quality
    features critical for dysarthria detection.

All features come back as a flat dict[str, float] so downstream stages can
treat them uniformly. NaN-safe (returns 0.0 for empty/silent segments).
"""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np

from FinalProject.shared.audio_io import TARGET_SR, to_numpy

warnings.filterwarnings("ignore", category=UserWarning)


def _safe(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


# ── librosa-based features ──────────────────────────────────────────────
def librosa_features(waveform, sr: int = TARGET_SR) -> Dict[str, float]:
    import librosa

    y = to_numpy(waveform) if hasattr(waveform, "detach") else np.asarray(waveform, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    duration = float(len(y)) / sr
    if duration < 0.05 or np.max(np.abs(y)) < 1e-4:
        return {"duration_s": duration}

    # Pitch via pyin (voiced-segment mean / std)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
        sr=sr, frame_length=2048,
    )
    f0_voiced = f0[voiced_flag] if f0 is not None else np.array([])
    f0_voiced = f0_voiced[np.isfinite(f0_voiced)] if f0_voiced.size else f0_voiced

    # RMS energy
    rms = librosa.feature.rms(y=y).flatten()

    # MFCC stats (13 dim, mean/std summarized)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Spectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).flatten()
    zcr = librosa.feature.zero_crossing_rate(y).flatten()

    # Pause structure: voiced/silent split via librosa.effects.split
    intervals = librosa.effects.split(y, top_db=30)
    voiced_dur = float(sum((b - a) for a, b in intervals)) / sr
    silent_dur = max(duration - voiced_dur, 0.0)
    n_pauses = max(0, len(intervals) - 1)
    longest_pause = 0.0
    if n_pauses > 0:
        gaps = [(intervals[i + 1][0] - intervals[i][1]) / sr
                for i in range(n_pauses)]
        longest_pause = float(max(gaps)) if gaps else 0.0

    return {
        "duration_s": duration,
        "voiced_dur_s": voiced_dur,
        "silent_dur_s": silent_dur,
        "n_pauses": float(n_pauses),
        "longest_pause_s": longest_pause,
        "pause_ratio": _safe(silent_dur / duration if duration else 0),
        "f0_mean_hz": _safe(f0_voiced.mean()) if f0_voiced.size else 0.0,
        "f0_std_hz": _safe(f0_voiced.std()) if f0_voiced.size else 0.0,
        "f0_range_hz": _safe(f0_voiced.max() - f0_voiced.min()) if f0_voiced.size else 0.0,
        "voiced_frame_ratio": _safe(voiced_flag.sum() / len(voiced_flag) if voiced_flag is not None and len(voiced_flag) else 0),
        "rms_mean": _safe(rms.mean()) if rms.size else 0.0,
        "rms_std": _safe(rms.std()) if rms.size else 0.0,
        "mfcc_mean_norm": _safe(np.linalg.norm(mfcc.mean(axis=1))),
        "mfcc_std_mean": _safe(mfcc.std(axis=1).mean()),
        "spec_centroid_mean": _safe(centroid.mean()) if centroid.size else 0.0,
        "spec_bandwidth_mean": _safe(bandwidth.mean()) if bandwidth.size else 0.0,
        "zcr_mean": _safe(zcr.mean()) if zcr.size else 0.0,
    }


# ── parselmouth (Praat) voice-quality features ──────────────────────────
def praat_features(waveform, sr: int = TARGET_SR) -> Dict[str, float]:
    import parselmouth
    from parselmouth.praat import call

    y = to_numpy(waveform) if hasattr(waveform, "detach") else np.asarray(waveform, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    if len(y) < int(0.05 * sr) or np.max(np.abs(y)) < 1e-4:
        return {}

    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)

    # PointProcess from cross-correlation pitch (75-500 Hz typical adult range)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter_local = call(point_process, "Get jitter (local)",
                        0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)",
                      0, 0, 0.0001, 0.02, 1.3)

    shimmer_local = call([snd, point_process], "Get shimmer (local)",
                         0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq11 = call([snd, point_process], "Get shimmer (apq11)",
                         0, 0, 0.0001, 0.02, 1.3, 1.6)

    # HNR via Praat's Harmonicity
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_mean = call(harmonicity, "Get mean", 0, 0)

    # Formants (mean of F1/F2/F3 across the utterance)
    formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    f1_mean = call(formant, "Get mean", 1, 0, 0, "hertz")
    f2_mean = call(formant, "Get mean", 2, 0, 0, "hertz")
    f3_mean = call(formant, "Get mean", 3, 0, 0, "hertz")

    return {
        "jitter_local": _safe(jitter_local),
        "jitter_rap": _safe(jitter_rap),
        "shimmer_local": _safe(shimmer_local),
        "shimmer_apq11": _safe(shimmer_apq11),
        "hnr_mean_db": _safe(hnr_mean),
        "f1_mean_hz": _safe(f1_mean),
        "f2_mean_hz": _safe(f2_mean),
        "f3_mean_hz": _safe(f3_mean),
    }


def all_acoustic_features(waveform, sr: int = TARGET_SR) -> Dict[str, float]:
    feats = librosa_features(waveform, sr=sr)
    feats.update(praat_features(waveform, sr=sr))
    return feats
