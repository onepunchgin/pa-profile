"""Pipeline 1 config: paths, model registry, thresholds.

Pipeline 1 = baseline SSD screener using SPRING lab finetuned ASR + MFA forced
alignment + acoustic features + frozen SSL embeddings + heuristic SSD scoring.

Models are registered as a dict so the pipeline can switch ASR backend without
code changes (we ablate to pick a winner, then lock it in).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch


def _envpath(env: str, default) -> Path:
    """Path overridable by env var; falls back to the workstation layout."""
    return Path(os.environ.get(env, str(default)))


# ── Repository roots ──────────────────────────────────────────────────────
LAB7_ROOT = _envpath("PA_PROFILE_LAB7_ROOT", "/media/csedept/lab7")
PROJECT_ROOT = _envpath("PA_PROFILE_PROJECT_ROOT", LAB7_ROOT / "FinalProject")
PIPELINE1_DIR = PROJECT_ROOT / "SpeechProfiling" / "pipeline1_baseline"
RUNS_DIR = _envpath("PA_PROFILE_RUNS_DIR", PIPELINE1_DIR / "runs")
try:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# ── Fairseq dict & user-dir for data2vec architecture registration ───────
SPRING_DICT_PATH = _envpath("PA_PROFILE_KANNADA_DICT", LAB7_ROOT / "SPRING_INX_Kannada_dict.txt")
DATA2VEC_USER_DIR = _envpath("PA_PROFILE_DATA2VEC_USERDIR", LAB7_ROOT / "FinetunedModels" / "data2vec_userdir")

# ── Audio ──
TARGET_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class ASRSpec:
    """Registry entry for an ASR backend.

    family: "wav2vec2" | "ccc_wav2vec2" | "hubert" | "data2vec"
    needs_user_dir: True for data2vec architectures (custom @register).
    needs_w2v_path_override: True for SSL-finetuned wav2vec2-style checkpoints
        that reference a base SSL model path inside the cfg.
    """
    name: str
    checkpoint: Path
    family: str
    dict_path: Path = SPRING_DICT_PATH
    needs_user_dir: bool = False
    needs_w2v_path_override: bool = True
    notes: str = ""


# Pipeline 1 candidates: SPRING lab's pre-finetuned Kannada ASR checkpoints.
# We will ablate over these and lock in the winner.
SPRING_FINETUNED: Dict[str, ASRSpec] = {
    "ccc_wav2vec2_kn": ASRSpec(
        name="ccc_wav2vec2_kn",
        checkpoint=LAB7_ROOT / "SPRING_INX_ccc_wav2vec2_Kannada.pt",
        family="ccc_wav2vec2",
        needs_w2v_path_override=True,
        notes="SPRING lab continuous-contrastive-coding wav2vec2 finetuned on Kannada",
    ),
    "data2vec_kn": ASRSpec(
        name="data2vec_kn",
        checkpoint=_envpath("PA_PROFILE_KANNADA_ASR_CKPT",
                            LAB7_ROOT / "SPRING_INX_data2vec_Kannada.pt"),
        family="data2vec",
        needs_user_dir=True,
        needs_w2v_path_override=False,
        notes="SPRING lab data2vec finetuned on Kannada",
    ),
    "hubert_kn": ASRSpec(
        name="hubert_kn",
        checkpoint=LAB7_ROOT / "SPRING_INX_HuBERT_Kannada.pt",
        family="hubert",
        needs_w2v_path_override=True,
        notes="SPRING lab HuBERT finetuned on Kannada",
    ),
}

# Pipeline 2 will reuse user's own checkpoints; included here for cross-ref.
# Each user finetune was trained against a different fairseq vocab/dict, so
# `dict_path` MUST match what was used at training time (otherwise CTC
# decoding produces gibberish).
_OPTIONB_DICT = LAB7_ROOT / "FinetunedModels/optionB_manifest/dict.ltr.txt"
_MIXED_DICT = LAB7_ROOT / "FinetunedModels/Mixed_manifest/dict.ltr.txt"

USER_FINETUNED: Dict[str, ASRSpec] = {
    "user_data2vec_kn": ASRSpec(
        name="user_data2vec_kn",
        checkpoint=LAB7_ROOT
        / "FinetunedModels/SPRING/data2vec_kannada_finetuned/checkpoint_best.pt",
        family="data2vec",
        dict_path=_OPTIONB_DICT,
        needs_user_dir=True,
        needs_w2v_path_override=False,
        notes="User's data2vec Kannada-only finetune (best WER ~17.22% on MILE; trained with optionB_manifest dict, 99 chars)",
    ),
    "user_data2vec_mixed": ASRSpec(
        name="user_data2vec_mixed",
        checkpoint=LAB7_ROOT
        / "FinetunedModels/SPRING/data2vec_mixed_training/checkpoint.best_wer_19.0662.pt",
        family="data2vec",
        dict_path=_MIXED_DICT,
        needs_user_dir=True,
        needs_w2v_path_override=False,
        notes="User's data2vec mixed-data finetune (Mixed_manifest dict, 99 chars)",
    ),
    "user_ssl_wav2vec2": ASRSpec(
        name="user_ssl_wav2vec2",
        checkpoint=LAB7_ROOT
        / "FinetunedModels/SPRING/ssl_kannada_finetuned/checkpoint_best.pt",
        family="ccc_wav2vec2",
        dict_path=_OPTIONB_DICT,
        notes="User's wav2vec2/SSL finetune (trained on MILE → optionB dict)",
    ),
}

# SSL-only encoder used for embeddings (NOT for transcription).
# data2vec-aqc is pretrained on Kannada — strong feature extractor.
SSL_FEATURE_EXTRACTOR = ASRSpec(
    name="data2vec_aqc_kn_ssl",
    checkpoint=LAB7_ROOT / "SPRING_INX_data2vec_aqc_Kannada.pt",
    family="data2vec",
    needs_user_dir=True,
    needs_w2v_path_override=True,
    notes="SPRING lab data2vec-aqc Kannada SSL pretrained (no CTC head)",
)


# Pipeline 1's locked-in ASR (winner of the 3-model ablation; see
# runs/ablation_20260502_101548.json — WER 11.81% / CER 1.36% on N=20 MILE).
DEFAULT_ASR_KEY = "data2vec_kn"


def all_models() -> Dict[str, ASRSpec]:
    out = dict(SPRING_FINETUNED)
    out.update(USER_FINETUNED)
    out["data2vec_aqc_kn_ssl"] = SSL_FEATURE_EXTRACTOR
    return out
