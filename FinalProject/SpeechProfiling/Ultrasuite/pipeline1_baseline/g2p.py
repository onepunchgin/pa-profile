"""Reference grapheme-to-phoneme for English child speech.

Strategy: word-level lookup in the UltraSuite-provided lexicon
(`uxssd.lex`, 1048 words). Words not in the lexicon fall back to a
quick rule-based approximation (vowel-consonant pattern preserved as
ARPA-ish phones) — enough for syllable counts and downstream feature
arithmetic when an unseen word appears.

Returns:
  text_to_words(text)     -> List[str]
  text_to_phones(text)    -> List[str]   (per-word phones concatenated)
  text_to_syllables(text) -> List[List[str]]  (one inner list per word)
"""
from __future__ import annotations

import re
from typing import List

from .data import load_lexicon

_LEX_CACHE = None
_VOWEL_PHONES = {
    # ARPAbet vowels (with stress digit suffix variants)
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
    # uxssd.lex variant (lowercase + non-ARPA chars like @)
    "a", "e", "i", "o", "u", "@", "I", "E", "U",
}


def _lex():
    global _LEX_CACHE
    if _LEX_CACHE is None:
        _LEX_CACHE = load_lexicon()
    return _LEX_CACHE


def _norm_word(w: str) -> str:
    return re.sub(r"[^A-Z']", "", w.upper())


def text_to_words(text: str) -> List[str]:
    return [w for w in (_norm_word(t) for t in text.split()) if w]


def _word_phones(word: str) -> List[str]:
    lex = _lex()
    if word in lex:
        return list(lex[word])
    # Fallback: emit one phone per char (so syllable counts approximate)
    return [c.lower() for c in word]


def text_to_phones(text: str) -> List[str]:
    out: List[str] = []
    for w in text_to_words(text):
        out.extend(_word_phones(w))
    return out


def _is_vowel(p: str) -> bool:
    bare = re.sub(r"\d", "", p)  # strip ARPAbet stress digit
    return bare in _VOWEL_PHONES or any(c in _VOWEL_PHONES for c in bare)


def text_to_syllables(text: str) -> List[List[str]]:
    """Vowel-nucleus heuristic: each vowel anchors a syllable, with the
    consonants between vowels apportioned by maximal-onset rule (simplified
    here to: consonants attach to the following vowel, except trailing
    consonants which join the previous syllable)."""
    words: List[List[str]] = []
    for w in text_to_words(text):
        phones = _word_phones(w)
        sylls: List[List[str]] = [[]]
        seen_vowel = False
        for p in phones:
            if _is_vowel(p):
                if seen_vowel:
                    sylls.append([])
                sylls[-1].append(p)
                seen_vowel = True
            else:
                sylls[-1].append(p)
        sylls = [s for s in sylls if s]
        words.append(sylls)
    return words


def n_syllables(text: str) -> int:
    return sum(len(s) for s in text_to_syllables(text))


def text_to_chars(text: str) -> List[str]:
    """Letter-level char list (for char-error-rate compatibility)."""
    return [c for c in re.sub(r"[^a-zA-Z]", "", text).lower()]
