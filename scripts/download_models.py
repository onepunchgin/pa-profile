"""Cross-platform model downloader (Windows-friendly alternative to download_models.sh).

Usage:
    python scripts/download_models.py
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(os.environ.get(
    "PA_PROFILE_ROOT", str(Path(__file__).resolve().parent.parent)
))
MODELS = ROOT / "models"

HF_USER = os.environ.get("PA_PROFILE_HF_USER", "onepunchgin")

REPOS = {
    "kannada_asr":  f"{HF_USER}/pa-profile-kannada-asr",
    "kannada_mfa":  f"{HF_USER}/pa-profile-kannada-mfa",
    "english_asr":  f"{HF_USER}/pa-profile-uxtd-wav2vec2",
    "stage8":       f"{HF_USER}/pa-profile-stage8-classifier",
}

for subdir, repo in REPOS.items():
    dst = MODELS / subdir
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[pa-profile] {repo} → {dst}")
    snapshot_download(repo_id=repo, local_dir=str(dst),
                      local_dir_use_symlinks=False)

print(f"[pa-profile] done. Set PA_PROFILE_ROOT={ROOT} before running app.py.")
