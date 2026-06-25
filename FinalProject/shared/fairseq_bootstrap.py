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

import sys
from pathlib import Path

FAIRSEQ_ROOT = Path("/home/prouser1/fairseq_inference/fairseq-0.12.2")
DATA2VEC_USER_DIR = Path("/media/csedept/lab7/FinetunedModels/data2vec_userdir")

_BOOTSTRAPPED = False
_USER_DIR_REGISTERED = False


def bootstrap_fairseq() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    p = str(FAIRSEQ_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    _BOOTSTRAPPED = True


def register_data2vec_userdir() -> None:
    global _USER_DIR_REGISTERED
    if _USER_DIR_REGISTERED:
        return
    bootstrap_fairseq()
    p = str(DATA2VEC_USER_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    import data2vec_ctc  # noqa: F401  -- triggers @register decorators
    _USER_DIR_REGISTERED = True
