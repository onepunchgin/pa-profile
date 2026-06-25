"""Canonical audio loading for the FinalProject pipelines.

All pipelines should load audio through `load_audio` so resampling, mono
conversion, and dtype handling stay consistent across stages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torchaudio

PathLike = Union[str, Path]
TARGET_SR = 16000


def load_audio(path: PathLike, target_sr: int = TARGET_SR) -> Tuple[torch.Tensor, int]:
    """Load `path`, resample to `target_sr`, downmix to mono.

    Returns (waveform, sr) where waveform shape is (T,) float32 in [-1, 1].
    """
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        sr = target_sr
    return waveform.squeeze(0).contiguous(), sr


def to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return waveform.detach().cpu().numpy().astype(np.float32, copy=False)


def duration_seconds(waveform: torch.Tensor, sr: int = TARGET_SR) -> float:
    return float(waveform.shape[-1]) / sr
