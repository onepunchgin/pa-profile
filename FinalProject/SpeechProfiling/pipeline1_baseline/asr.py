"""ASR adapter — uniform interface over SPRING fairseq ASR checkpoints.

Mirrors the loading pattern in
  /media/csedept/lab7/FinetunedModels/SPRING/optionB_baseline.py
and
  /media/csedept/lab7/FinetunedModels/SPRING/data2vec/step4b_evaluate_finetuned.py
so any model registered in `config.SPRING_FINETUNED` or `USER_FINETUNED` loads
identically and exposes a single `transcribe(wav_path)` API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Bootstrap fairseq's namespace-package examples BEFORE any fairseq import.
from FinalProject.shared.fairseq_bootstrap import (
    bootstrap_fairseq,
    register_data2vec_userdir,
)

bootstrap_fairseq()

import torch
import torchaudio

from .config import ASRSpec, DEVICE, TARGET_SR


def _load_vocab(dict_path: Path) -> dict[int, str]:
    """fairseq reserves ids 0..3 for special tokens, dict.ltr starts at 4."""
    vocab: dict[int, str] = {}
    with open(dict_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            tok = line.strip().split()
            if not tok:
                continue
            vocab[idx + 4] = tok[0]
    return vocab


def _ctc_decode(pred_ids, vocab: dict[int, str]) -> str:
    """Collapse repeats and blanks (id=0), map to chars, replace `|` -> space."""
    decoded: list[int] = []
    prev = -1
    for tok in pred_ids:
        tok_i = int(tok)
        if tok_i != prev and tok_i != 0:
            decoded.append(tok_i)
        prev = tok_i
    text = "".join(vocab.get(i, "") for i in decoded)
    return text.replace("|", " ").strip()


class ASRModel:
    """Thin wrapper around a loaded fairseq ASR model + its vocab."""

    def __init__(self, spec: ASRSpec, model, vocab: dict[int, str]):
        self.spec = spec
        self.model = model
        self.vocab = vocab

    @torch.no_grad()
    def logits(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return CTC logits (T, vocab) for a 1D waveform tensor."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(DEVICE)
        padding_mask = torch.zeros(waveform.shape, dtype=torch.bool, device=DEVICE)
        out = self.model(source=waveform, padding_mask=padding_mask)
        if "encoder_out" in out:
            return out["encoder_out"].squeeze(1)
        if "x" in out:
            return out["x"].squeeze(0)
        raise ValueError(f"Unknown model output keys: {list(out.keys())}")

    def transcribe_tensor(self, waveform: torch.Tensor) -> str:
        logits = self.logits(waveform)
        pred = logits.argmax(dim=-1).cpu().numpy()
        return _ctc_decode(pred, self.vocab)

    def transcribe(self, wav_path) -> str:
        from FinalProject.shared.audio_io import load_audio  # type: ignore
        waveform, _ = load_audio(wav_path, target_sr=TARGET_SR)
        return self.transcribe_tensor(waveform)


def load_asr(spec: ASRSpec) -> ASRModel:
    """Load an ASR checkpoint per its family-specific knobs.

    Notes on family quirks:
    - data2vec finetuned checkpoints drop SSL-only modules (quantizer,
      project_q, contr_proj) → must load with strict=False.
    - HuBERT task config carries `label_dir` from SPRING cluster paths that
      don't exist locally → override it to a benign existing dir; inference
      path doesn't actually read labels.
    """
    import fairseq.checkpoint_utils

    if spec.needs_user_dir:
        register_data2vec_userdir()

    arg_overrides: dict = {"data": str(spec.dict_path.parent)}
    if spec.needs_w2v_path_override:
        arg_overrides["w2v_path"] = str(spec.checkpoint)
    if spec.family == "hubert":
        # HuBERT's task cfg references a label_dir on the SPRING cluster
        # plus a `labels: ['km']` cluster dict. Point it at our stub dir
        # which contains a placeholder dict.km.txt (never read at inference).
        arg_overrides["label_dir"] = str(
            Path(__file__).parent / "_assets"
        )

    print(f"[asr] loading {spec.name} from {spec.checkpoint}")
    print(f"[asr]   family={spec.family}  dict={spec.dict_path}")
    models, _cfg, _task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [str(spec.checkpoint)],
        arg_overrides=arg_overrides,
        strict=False,
    )
    model = models[0].to(DEVICE).eval()
    vocab = _load_vocab(spec.dict_path)
    print(f"[asr]   loaded; vocab size (incl specials) = {len(vocab) + 4}")
    return ASRModel(spec=spec, model=model, vocab=vocab)
