"""English child-speech ASR adapter for UltraSuite.

Default model: `facebook/wav2vec2-base-960h` (adult LibriSpeech, but well
packaged + standard CTC). Adult ASR will under-transcribe child speech
(expect 30-50 % WER on UXSSD children); Phase 2 of the Ultrasuite plan
fine-tunes this on child data (UXSSD or MyST) to fix.

Considered alternatives:
- `lijialudew/wav2vec_children_ASR` — explicitly child-tuned but packaged
  as a SpeechBrain checkpoint dump (not a standard HF model). Would need
  a SpeechBrain loader; left as a Phase 2 follow-up.
- `openai/whisper-small.en` — strong on children, but seq2seq instead of
  CTC; would need a separate transcribe-only path. Doable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16_000
# UXTD-fine-tuned (Phase 2) is the default — confirmed materially better than
# the adult baseline on UXSSD child speech (smoke test 2026-05-06: baseline
# transcribed 'umbrella' as 'unduella' and 'train' as 'twain'; the fine-tune
# returns 'elephant umbrella train swin'). Pass `hf_model="facebook/wav2vec2-base-960h"`
# to compare against the un-finetuned baseline.
DEFAULT_HF_MODEL = os.environ.get(
    "PA_PROFILE_ENGLISH_ASR",
    "/media/csedept/lab7/FinalProject/SpeechProfiling/Ultrasuite/pipeline2_finetuned/runs/uxtd_v1/final",
)

_cache: dict = {}


@dataclass
class EnglishASRModel:
    name: str
    model: Wav2Vec2ForCTC
    processor: Wav2Vec2Processor
    vocab: Dict[int, str]   # {token_id: token_str}, includes specials

    @torch.no_grad()
    def logits(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return (T_enc, V) CTC logits for a 1-D waveform tensor at 16kHz."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # Processor handles normalisation
        inputs = self.processor(waveform.squeeze().cpu().numpy(),
                                sampling_rate=TARGET_SR,
                                return_tensors="pt")
        input_values = inputs.input_values.to(DEVICE)
        out = self.model(input_values).logits  # (1, T, V)
        return out.squeeze(0)

    @torch.no_grad()
    def transcribe_tensor(self, waveform: torch.Tensor) -> str:
        logits = self.logits(waveform)
        pred_ids = logits.argmax(dim=-1).unsqueeze(0)
        return self.processor.batch_decode(pred_ids)[0].strip()

    def transcribe(self, wav_path) -> str:
        waveform, sr = torchaudio.load(str(wav_path))
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        return self.transcribe_tensor(waveform.mean(dim=0))


def load_english_asr(hf_model: str = DEFAULT_HF_MODEL) -> EnglishASRModel:
    if hf_model in _cache:
        return _cache[hf_model]
    print(f"[asr-en] loading {hf_model} from HuggingFace …")
    processor = Wav2Vec2Processor.from_pretrained(hf_model)
    model = Wav2Vec2ForCTC.from_pretrained(hf_model).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    vocab = {v: k for k, v in processor.tokenizer.get_vocab().items()}
    print(f"[asr-en]   vocab size = {len(vocab)}")
    asr = EnglishASRModel(name=hf_model, model=model,
                          processor=processor, vocab=vocab)
    _cache[hf_model] = asr
    return asr
