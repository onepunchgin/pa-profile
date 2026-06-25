"""Reference-vs-hypothesis comparison at multiple granularities.

Produces structured error patterns that the SSD scoring stage maps to
disorder categories. Granularities computed:
  - word level (WER + sub/del/ins counts + words affected)
  - syllable level (rule-based Kannada syllabifier)
  - character level (Kannada-only chars)
  - phoneme level (espeak IPA — approximate for Kannada)
  - phoneme-class patterns (retroflex<->dental, aspirate deaspiration,
    fricative, vowel-length, gemination) — these are the SSD-relevant signals
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from FinalProject.shared.evaluation import edit_ops
from FinalProject.shared.text_norm import (
    APPROXIMANT,
    ASPIRATED,
    DENTAL,
    FRICATIVE,
    LONG_VOWEL_SIGNS,
    LONG_VOWELS,
    NASAL,
    RETROFLEX,
    SHORT_VOWEL_SIGNS,
    SHORT_VOWELS,
    UNASPIRATED,
    is_consonant,
    normalize,
)

from .g2p import text_to_chars, text_to_phonemes, text_to_syllables


@dataclass
class LevelStats:
    n_ref: int
    n_hyp: int
    substitutions: int
    deletions: int
    insertions: int
    @property
    def error_rate(self) -> float:
        return (self.substitutions + self.deletions + self.insertions) / max(self.n_ref, 1)


@dataclass
class PatternCounts:
    """Tallies of SSD-relevant error patterns extracted from ref vs hyp chars."""
    retroflex_to_dental: int = 0      # ಟ→ತ etc — articulation / phonological
    dental_to_retroflex: int = 0
    deaspiration: int = 0             # ಖ→ಕ etc — common in CAS, articulation
    aspiration_added: int = 0
    fricative_substitution: int = 0   # /s/ /ʃ/ /ʂ/ disturbed — articulation
    vowel_length_short_for_long: int = 0  # ಆ→ಅ — CAS / dysarthria
    vowel_length_long_for_short: int = 0
    geminate_simplified: int = 0      # ಲ್ಲ→ಲ — phonological (cluster reduction)
    final_consonant_deletion: int = 0 # ends-with consonant in ref but not hyp
    nasal_substitution: int = 0
    approximant_substitution: int = 0


@dataclass
class ComparisonResult:
    word: LevelStats
    syllable: LevelStats
    char: LevelStats
    phoneme: LevelStats
    pattern: PatternCounts
    aligned_pairs: List[tuple] = field(default_factory=list)  # (ref_ch, hyp_ch_or_None)


def _level_stats(ref: Sequence, hyp: Sequence) -> LevelStats:
    s, d, i, n = edit_ops(ref, hyp)
    return LevelStats(n_ref=n, n_hyp=len(hyp), substitutions=s, deletions=d, insertions=i)


def _aligned_pairs(ref: List[str], hyp: List[str]) -> List[tuple]:
    """Per-ref-position pair (ref_ch, hyp_ch_or_None) via Levenshtein backtrace."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    out = []
    i, j = n, m
    while i > 0:
        if j > 0 and ref[i - 1] == hyp[j - 1]:
            out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        else:
            sub = dp[i - 1][j - 1] if j > 0 else 10**9
            dele = dp[i - 1][j]
            ins = dp[i][j - 1] if j > 0 else 10**9
            best = min(sub, dele, ins)
            if best == sub and j > 0:
                out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
            elif best == dele:
                out.append((ref[i - 1], None)); i -= 1
            else:
                j -= 1  # spurious insertion in hyp; ignore for ref-aligned pairs
    out.reverse()
    return out


def _classify_patterns(pairs: List[tuple], ref_chars: List[str]) -> PatternCounts:
    pc = PatternCounts()
    for r, h in pairs:
        if h is None:
            # deletion of a ref char
            if is_consonant(r) and r == ref_chars[-1]:
                pc.final_consonant_deletion += 1
            continue
        if r == h:
            continue
        # substitutions
        if r in RETROFLEX and h in DENTAL:
            pc.retroflex_to_dental += 1
        elif r in DENTAL and h in RETROFLEX:
            pc.dental_to_retroflex += 1
        if r in ASPIRATED and h in UNASPIRATED:
            pc.deaspiration += 1
        elif r in UNASPIRATED and h in ASPIRATED:
            pc.aspiration_added += 1
        if r in FRICATIVE and h != r:
            pc.fricative_substitution += 1
        if r in NASAL and h not in NASAL:
            pc.nasal_substitution += 1
        if r in APPROXIMANT and h not in APPROXIMANT and h != r:
            pc.approximant_substitution += 1
        if r in LONG_VOWELS and h in SHORT_VOWELS:
            pc.vowel_length_short_for_long += 1
        elif r in SHORT_VOWELS and h in LONG_VOWELS:
            pc.vowel_length_long_for_short += 1
        if r in LONG_VOWEL_SIGNS and h in SHORT_VOWEL_SIGNS:
            pc.vowel_length_short_for_long += 1
        elif r in SHORT_VOWEL_SIGNS and h in LONG_VOWEL_SIGNS:
            pc.vowel_length_long_for_short += 1

    # Geminate simplification: detect doubled consonant in ref that's single in hyp
    ref_str = "".join(ref_chars)
    for i in range(len(ref_chars) - 2):
        if (is_consonant(ref_chars[i]) and ord(ref_chars[i + 1]) == 0x0CCD
                and ref_chars[i + 2] == ref_chars[i]):
            # geminate pattern in ref; check if hyp lost it
            # crude: if number of that char in hyp < ref, count one
            r_ch = ref_chars[i]
            ref_count = sum(1 for r, _ in pairs if r == r_ch)
            hyp_count = sum(1 for _, h in pairs if h == r_ch)
            if hyp_count < ref_count:
                pc.geminate_simplified += 1

    return pc


def compare(reference_text: str, hypothesis_text: str) -> ComparisonResult:
    ref_text = normalize(reference_text)
    hyp_text = normalize(hypothesis_text)

    ref_words = ref_text.split()
    hyp_words = hyp_text.split()

    ref_sylls = text_to_syllables(ref_text)
    hyp_sylls = text_to_syllables(hyp_text)

    ref_chars = text_to_chars(ref_text)
    hyp_chars = text_to_chars(hyp_text)

    # phonemes — espeak; can be empty if backend missing
    try:
        ref_phones = text_to_phonemes(ref_text)
        hyp_phones = text_to_phonemes(hyp_text)
    except Exception:
        ref_phones, hyp_phones = [], []

    pairs = _aligned_pairs(ref_chars, hyp_chars)
    patterns = _classify_patterns(pairs, ref_chars)

    return ComparisonResult(
        word=_level_stats(ref_words, hyp_words),
        syllable=_level_stats(ref_sylls, hyp_sylls),
        char=_level_stats(ref_chars, hyp_chars),
        phoneme=_level_stats(ref_phones, hyp_phones),
        pattern=patterns,
        aligned_pairs=pairs,
    )
