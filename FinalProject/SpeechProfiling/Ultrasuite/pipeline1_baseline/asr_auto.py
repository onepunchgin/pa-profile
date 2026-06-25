"""Backbone-agnostic ASR loader for the §5.3 replication experiment.

Mirrors `asr.py::load_english_asr` but dispatches via `AutoModelForCTC`
so any wav2vec-2.0-family backbone — wav2vec 2.0, HuBERT, WavLM, XLS-R,
XLSR-53 — can be loaded with a single call. The processor is forced to
the LibriSpeech-style character vocab from wav2vec2-base-960h so the
post-CTC text outputs are directly comparable across backbones.

`load_english_asr_auto(model_dir)` returns an EnglishASRModel object
that's drop-in compatible with the v2 feature extractor.
"""
from __future__ import annotations

import torch
import torchaudio
from transformers import AutoModelForCTC, Wav2Vec2Processor

from .asr import EnglishASRModel, TARGET_SR, DEVICE

PROCESSOR_SOURCE = "facebook/wav2vec2-base-960h"

_cache_auto: dict = {}


def load_english_asr_auto(model_dir: str,
                           processor_source: str = PROCESSOR_SOURCE
                           ) -> EnglishASRModel:
    key = (model_dir, processor_source)
    if key in _cache_auto:
        return _cache_auto[key]
    print(f"[asr-auto] loading backbone: {model_dir}")
    print(f"[asr-auto] using processor: {processor_source}")
    processor = Wav2Vec2Processor.from_pretrained(processor_source)
    model = AutoModelForCTC.from_pretrained(model_dir).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    vocab = {v: k for k, v in processor.tokenizer.get_vocab().items()}
    print(f"[asr-auto]   vocab size = {len(vocab)}")
    asr = EnglishASRModel(name=model_dir, model=model,
                          processor=processor, vocab=vocab)
    _cache_auto[key] = asr
    return asr
