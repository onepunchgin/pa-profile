#!/usr/bin/env bash
# Pull all PA-Profile model weights from Hugging Face Hub into ./models/.
#
# Requires:
#   pip install huggingface_hub
#   (optional) export HF_TOKEN=hf_xxx   # for private repos
#
# Usage:
#   bash scripts/download_models.sh
#
# Override the destination with PA_PROFILE_ROOT=/some/path bash scripts/...

set -euo pipefail

ROOT="${PA_PROFILE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODELS="$ROOT/models"
mkdir -p "$MODELS"/{kannada_asr,kannada_mfa,english_asr,stage8}

# CHANGE THESE if you forked the repos under your own HF username:
HF_USER="${PA_PROFILE_HF_USER:-onepunchgin}"

echo "[pa-profile] downloading model weights into: $MODELS"

# Use huggingface-cli when available; fall back to the Python API.
if command -v huggingface-cli >/dev/null 2>&1; then
    DL() { huggingface-cli download "$1" --local-dir "$2" --local-dir-use-symlinks False; }
else
    DL() {
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$1', local_dir='$2', local_dir_use_symlinks=False)
"
    }
fi

# Kannada ASR (3.76 GB) — usually a PRIVATE repo (SPRING licence). Needs HF_TOKEN.
DL "${HF_USER}/pa-profile-kannada-asr" "$MODELS/kannada_asr"

# Kannada MFA acoustic model + pron dict (~62 MB) — yours, public.
DL "${HF_USER}/pa-profile-kannada-mfa" "$MODELS/kannada_mfa"

# English UXTD-finetuned wav2vec2 (~361 MB) — yours, public.
DL "${HF_USER}/pa-profile-uxtd-wav2vec2" "$MODELS/english_asr"

# Stage-8 SSD-vs-TD classifier (<5 MB) — yours, public.
DL "${HF_USER}/pa-profile-stage8-classifier" "$MODELS/stage8"

echo "[pa-profile] done. Set PA_PROFILE_ROOT=$ROOT before running app.py."
