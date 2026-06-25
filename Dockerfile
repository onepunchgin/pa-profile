# PA-Profile portable runtime — Hugging Face Space (Docker SDK) compatible.
#
# Build:
#   docker build -t pa-profile .
# Run:
#   docker run --rm -p 7860:7860 pa-profile
#
# For private HF Hub repos (e.g. the SPRING Kannada checkpoint under a
# private repo) pass an HF token at build time:
#   docker build --build-arg HF_TOKEN=hf_xxx -t pa-profile .

FROM mambaorg/micromamba:1.5.8

# 1. System deps (Kaldi runtime libs needed by MFA, audio decoders, git-lfs
#    so the model-pull step can do LFS clones if the user prefers it).
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs build-essential libsndfile1 ffmpeg curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2. Python 3.10 + Montreal Forced Aligner (pulls Kaldi binaries +
#    PostgreSQL from conda-forge — the only sane way to install MFA).
RUN micromamba install -y -n base -c conda-forge \
        python=3.10 \
        montreal-forced-aligner=2.2.17 \
        postgresql && \
    micromamba clean --all -y

# 3. Pure-Python deps for the demo (torch / transformers / gradio / fairseq).
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt
RUN micromamba run -n base pip install --no-cache-dir -r /tmp/requirements.txt

# 4. Pre-download the English MFA acoustic + dictionary so the very first
#    request doesn't pay the ~70 MB download.
RUN micromamba run -n base mfa model download acoustic english_us_arpa && \
    micromamba run -n base mfa model download dictionary english_us_arpa

# 5. App code + bundled samples + models. On HF Spaces the LFS clone copies
#    the model weights as part of the build; for a leaner image use the
#    HF_HUB_PULL build arg to fetch them at build time instead (see below).
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# 6. Pull model weights from HF Hub at build time (they are NOT committed to
#    this repo). The private SPRING Kannada checkpoint needs an auth token,
#    supplied as a Space secret named HF_TOKEN and read via a BuildKit secret
#    mount so the token is never baked into an image layer.
ARG HF_HUB_PULL=1
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false \
    if [ "$HF_HUB_PULL" = "1" ]; then \
        export HF_TOKEN="$(cat /run/secrets/HF_TOKEN 2>/dev/null || true)" ; \
        micromamba run -n base bash /app/scripts/download_models.sh ; \
    fi

# 7. Runtime env — point the pipelines at the bundled models.
ENV PA_PROFILE_ROOT=/app
ENV PA_PROFILE_KANNADA_ASR_CKPT=/app/models/kannada_asr/SPRING_INX_data2vec_Kannada.pt
ENV PA_PROFILE_KANNADA_DICT=/app/models/kannada_asr/SPRING_INX_Kannada_dict.txt
ENV PA_PROFILE_DATA2VEC_USERDIR=/app/models/kannada_asr/data2vec_userdir
ENV PA_PROFILE_KANNADA_MFA_ZIP=/app/models/kannada_mfa/kannada_v2b.zip
ENV PA_PROFILE_KANNADA_PRON_DICT=/app/models/kannada_mfa/pron_dict.txt
ENV PA_PROFILE_ENGLISH_ASR=/app/models/english_asr
ENV PA_PROFILE_STAGE8_MODEL=/app/models/stage8/model.joblib
ENV PA_PROFILE_MFA_BIN=""

# HF Spaces injects PORT; default to 7860 for local Docker.
ENV PORT=7860
EXPOSE 7860

CMD ["micromamba", "run", "-n", "base", "python", "/app/app.py"]
