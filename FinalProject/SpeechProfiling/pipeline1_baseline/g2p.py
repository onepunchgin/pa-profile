"""Kannada G2P (grapheme-to-phoneme) for SSD profiling.

Two access points:
  - `text_to_phonemes(text)`: phonemizer (espeak-ng kn backend) -> list of IPA
    phoneme tokens for the whole utterance. Approximate but the best off-the-
    shelf option for low-resource Kannada.
  - `text_to_syllables(text)`: rule-based Kannada syllabifier (no external deps;
    self-contained in shared.text_norm).

We expose both because SSD analysis benefits from multiple granularities:
syllables for rate / fluency, phonemes for articulation / phonological
patterns.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from FinalProject.shared.text_norm import normalize, syllabify


@lru_cache(maxsize=1)
def _phonemizer():
    from phonemizer.backend import EspeakBackend
    return EspeakBackend(
        language="kn",
        preserve_punctuation=False,
        with_stress=False,
        words_mismatch="ignore",
    )


def text_to_phonemes(text: str) -> List[str]:
    """Return a flat list of IPA phoneme tokens for `text` (Kannada)."""
    text = normalize(text)
    if not text:
        return []
    out = _phonemizer().phonemize([text], strip=True)[0]
    # Espeak separates phonemes by space within a word and by ' ' between
    # words; normalize to a single flat list.
    return [tok for tok in out.replace("  ", " ").split() if tok]


def text_to_syllables(text: str) -> List[str]:
    return syllabify(normalize(text))


def text_to_chars(text: str) -> List[str]:
    """Kannada characters as a list, dropping whitespace."""
    return [ch for ch in normalize(text) if not ch.isspace()]
