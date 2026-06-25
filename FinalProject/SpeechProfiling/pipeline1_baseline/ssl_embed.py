"""Frozen SSL embedding from SPRING_INX_data2vec_aqc_Kannada.pt.

The aqc_Kannada checkpoint is SSL-pretrained on Kannada (no CTC head). We
load it via fairseq, pull the encoder output for an utterance, and pool
mean+std to a fixed-size descriptor that downstream stages (and a future
trained classifier) can consume.

This is intentionally shallow — just an utterance-level embedding —
because we don't have labels to fine-tune anything bigger right now.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from FinalProject.shared.audio_io import TARGET_SR
from FinalProject.shared.fairseq_bootstrap import (
    bootstrap_fairseq,
    register_data2vec_userdir,
)

bootstrap_fairseq()

from .config import DEVICE, SSL_FEATURE_EXTRACTOR


@dataclass
class _LoadedSSL:
    model: torch.nn.Module
    name: str


_cached: Optional[_LoadedSSL] = None


def _load_ssl() -> _LoadedSSL:
    global _cached
    if _cached is not None:
        return _cached
    import fairseq.checkpoint_utils
    register_data2vec_userdir()
    spec = SSL_FEATURE_EXTRACTOR
    arg_overrides = {"data": str(spec.dict_path.parent),
                     "w2v_path": str(spec.checkpoint)}
    print(f"[ssl] loading {spec.name}")
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [str(spec.checkpoint)], arg_overrides=arg_overrides, strict=False
    )
    model = models[0].to(DEVICE).eval()
    _cached = _LoadedSSL(model=model, name=spec.name)
    return _cached


@torch.no_grad()
def embed(waveform: torch.Tensor) -> dict:
    """Return mean + std pooled features over the encoder time axis.

    Output shape: {'mean': (D,), 'std': (D,), 'norm': scalar}
    """
    wrap = _load_ssl()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.to(DEVICE)
    padding_mask = torch.zeros(waveform.shape, dtype=torch.bool, device=DEVICE)

    # data2vec SSL: use extract_features (no decoder/quantizer head)
    out = wrap.model.extract_features(source=waveform, padding_mask=padding_mask)
    if isinstance(out, dict):
        # fairseq may return {"x": ..., "padding_mask": ...}
        x = out.get("x")
    elif isinstance(out, tuple):
        x = out[0]
    else:
        x = out
    if x is None:
        raise RuntimeError("SSL extract_features returned no tensor")
    # x shape is (B, T, D); pool over T
    feats = x.squeeze(0).float().cpu().numpy()
    if feats.ndim == 1:
        feats = feats[None, :]
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "norm": float(np.linalg.norm(mean)),
        "T": int(feats.shape[0]),
        "D": int(feats.shape[1]),
    }
