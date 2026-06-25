"""Bootstrap fairseq imports.

The local fairseq dev install at /home/prouser1/fairseq_inference/fairseq-0.12.2
references `examples.data2vec.models` (a namespace package) at import time.
That only resolves when the fairseq install root is on `sys.path`. Calling
`bootstrap_fairseq()` once before any fairseq import makes scripts portable
to any cwd.

Also registers the user's data2vec_userdir for `data2vec_ctc` etc. when
requested.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Portable: prefer env vars. HF Spaces / Docker set PA_PROFILE_DATA2VEC_USERDIR
# to the bundled model dir (models/kannada_asr/data2vec_userdir). pip-installed
# fairseq doesn't need FAIRSEQ_ROOT on sys.path, so a missing default is harmless.
FAIRSEQ_ROOT = Path(os.environ.get(
    "PA_PROFILE_FAIRSEQ_ROOT", "/home/prouser1/fairseq_inference/fairseq-0.12.2"))
DATA2VEC_USER_DIR = Path(os.environ.get(
    "PA_PROFILE_DATA2VEC_USERDIR", "/media/csedept/lab7/FinetunedModels/data2vec_userdir"))

# Vendored data2vec model code (examples.data2vec.models). pip fairseq 0.12.2
# ships these only under examples/ (not in the wheel), so we bundle them next to
# this module and put the dir on sys.path so `import examples.data2vec.models`
# resolves and registers the architecture the Kannada checkpoint needs.
_BUNDLED_FAIRSEQ_EXAMPLES = Path(__file__).resolve().parent / "_fairseq_examples"

_BOOTSTRAPPED = False
_USER_DIR_REGISTERED = False


def bootstrap_fairseq() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    p = str(FAIRSEQ_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    # Vendored examples as a fallback so `examples.data2vec.models` resolves on
    # the Space (where the workstation's fairseq checkout doesn't exist).
    pe = str(_BUNDLED_FAIRSEQ_EXAMPLES)
    if pe not in sys.path:
        sys.path.append(pe)
    _BOOTSTRAPPED = True


def register_data2vec_userdir() -> None:
    global _USER_DIR_REGISTERED
    if _USER_DIR_REGISTERED:
        return
    bootstrap_fairseq()
    # Register the base data2vec architectures first. On the workstation the
    # patched fairseq checkout auto-imports examples.data2vec.models; with pip
    # fairseq we import the vendored copy ourselves (no-op if already loaded).
    try:
        import examples.data2vec.models  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"[fairseq_bootstrap] examples.data2vec.models import: {exc}", flush=True)
    p = str(DATA2VEC_USER_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    import data2vec_ctc  # noqa: F401  -- triggers @register decorators
    _USER_DIR_REGISTERED = True
