"""Kannada text normalization using indic_nlp_library (vendored at
/media/csedept/lab7/SpeechProfiling/indic_nlp_library).
"""
from __future__ import annotations

import os
import re
import sys
from functools import lru_cache

INDIC_NLP_PATH = "/media/csedept/lab7/SpeechProfiling/indic_nlp_library"
KANNADA_RANGE = (0x0C80, 0x0CFF)
ZWS = "‌"  # zero-width non-joiner appears in Kannada transcripts
ZWJ = "‍"


def _ensure_indicnlp_on_path() -> None:
    if INDIC_NLP_PATH not in sys.path:
        sys.path.insert(0, INDIC_NLP_PATH)


@lru_cache(maxsize=1)
def _normalizer():
    _ensure_indicnlp_on_path()
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    return IndicNormalizerFactory().get_normalizer("kn")


def normalize(text: str) -> str:
    """Normalize Kannada text: trim, collapse whitespace, drop ZWJ/ZWNJ,
    keep only Kannada Unicode + spaces, then run indic_nlp normalizer."""
    text = text.replace(ZWS, "").replace(ZWJ, "")
    cleaned = []
    for ch in text:
        cp = ord(ch)
        if KANNADA_RANGE[0] <= cp <= KANNADA_RANGE[1] or ch.isspace():
            cleaned.append(ch)
    text = "".join(cleaned)
    text = re.sub(r"\s+", " ", text).strip()
    return _normalizer().normalize(text)


def chars_only_kannada(text: str) -> str:
    """Strip everything except Kannada code points (no whitespace)."""
    return "".join(ch for ch in text if KANNADA_RANGE[0] <= ord(ch) <= KANNADA_RANGE[1])


# Kannada character classes — used by SSD-relevant phoneme analysis.
# These are the consonants/vowels most frequently implicated in childhood
# Kannada speech-sound disorders per clinical literature (substitutions of
# retroflex→dental, deaspiration of aspirates, fricative distortion, length
# contrasts, geminate simplification).
CONSONANT_RANGE = (0x0C95, 0x0CB9)
VOWEL_INDEP = set(range(0x0C85, 0x0C95))
VOWEL_DEP = (set(range(0x0CBE, 0x0CC5)) | set(range(0x0CC6, 0x0CC9))
             | set(range(0x0CCA, 0x0CCD)))
HALANT = 0x0CCD
ANUSVARA_VISARGA = {0x0C82, 0x0C83}

RETROFLEX = set("ಟಠಡಢಣಷಳ")
DENTAL = set("ತಥದಧನ")
ASPIRATED = set("ಖಘಛಝಠಢಥಧಫಭ")
UNASPIRATED = set("ಕಗಚಜಟಡತದಪಬ")
FRICATIVE = set("ಶಷಸಹ")
NASAL = set("ಙಞಣನಮ")
APPROXIMANT = set("ಯರಲವಳ")
SHORT_VOWELS = set("ಅಇಉಋಎಒ")
LONG_VOWELS = set("ಆಈಊೠಏಓ")
SHORT_VOWEL_SIGNS = set("ಿುೃೆೊ")
LONG_VOWEL_SIGNS = set("ಾೀೂೄೇೋ")


def is_consonant(ch: str) -> bool:
    cp = ord(ch)
    return CONSONANT_RANGE[0] <= cp <= CONSONANT_RANGE[1] or cp == 0x0CDE


def syllabify(text: str) -> list[str]:
    """Rule-based Kannada (Brahmi-derived) orthographic syllabification.

    Avoids the indic_nlp_resources external dataset dependency. A syllable
    is a maximal run of {consonant + halant}* + (consonant | independent
    vowel) + (vowel-sign | anusvara | visarga | halant)*. Whitespace and
    non-Kannada chars terminate a syllable.
    """
    out: list[str] = []
    cur: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        cp = ord(ch)
        if not (KANNADA_RANGE[0] <= cp <= KANNADA_RANGE[1]):
            if cur:
                out.append("".join(cur)); cur = []
            i += 1
            continue
        is_indep_vowel = 0x0C85 <= cp <= 0x0C94
        is_consonant = 0x0C95 <= cp <= 0x0CB9 or cp == 0x0CDE
        is_dependent = (
            0x0CBE <= cp <= 0x0CC4 or 0x0CC6 <= cp <= 0x0CC8
            or 0x0CCA <= cp <= 0x0CCC or cp in (0x0CD5, 0x0CD6)
            or cp in (0x0C82, 0x0C83)
        )
        is_halant = cp == 0x0CCD
        if (is_indep_vowel or is_consonant) and cur and not (cur and ord(cur[-1]) == 0x0CCD):
            out.append("".join(cur)); cur = []
        cur.append(ch)
        i += 1
    if cur:
        out.append("".join(cur))
    return [s for s in out if s.strip()]
