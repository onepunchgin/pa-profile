#!/usr/bin/env bash
# One-time helper: push the locally-bundled model weights into four
# Hugging Face Hub model repos. Run this once from the workstation; after
# that, anyone else uses download_models.sh to pull them.
#
# Prereqs:
#   pip install huggingface_hub
#   huggingface-cli login   # paste an HF token with write scope
#
# Usage:
#   PA_PROFILE_HF_USER=onepunchgin bash scripts/push_to_hf_hub.sh

set -euo pipefail

if [ -z "${PA_PROFILE_HF_USER:-}" ]; then
    echo "ERROR: set PA_PROFILE_HF_USER to your Hugging Face username/org" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"

push_one() {
    local repo_short="$1"
    local local_dir="$2"
    local private_flag="$3"   # "--private" or ""
    local repo_full="${PA_PROFILE_HF_USER}/${repo_short}"

    echo
    echo "=== pushing $local_dir → $repo_full ==="
    huggingface-cli repo create "$repo_short" --type model $private_flag -y || true
    huggingface-cli upload "$repo_full" "$local_dir" . --repo-type model
}

# The SPRING Kannada checkpoint redistribution status is unclear — push as
# PRIVATE by default. The Space loads it via an HF_TOKEN secret.
push_one "pa-profile-kannada-asr"        "$MODELS/kannada_asr"   "--private"
push_one "pa-profile-kannada-mfa"        "$MODELS/kannada_mfa"   ""
push_one "pa-profile-uxtd-wav2vec2"      "$MODELS/english_asr"   ""
push_one "pa-profile-stage8-classifier"  "$MODELS/stage8"        ""

echo
echo "All four repos pushed. Update scripts/download_models.sh with your HF username."
