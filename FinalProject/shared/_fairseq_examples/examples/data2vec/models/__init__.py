"""Vendored data2vec model registrations.

pip-installed fairseq 0.12.2 ships the data2vec model definitions only under
``examples/`` (which is excluded from the wheel). The portable build vendors the
two model files here and imports them so fairseq registers the ``data2vec_audio``
architecture that the Kannada SPRING checkpoint needs. On the workstation the
patched fairseq checkout does this automatically via its models/__init__.py.
"""
from . import data2vec_audio  # noqa: F401  -- triggers @register_model
try:
    from . import data2vec_text  # noqa: F401  -- not needed for ASR, but harmless
except Exception:
    pass
