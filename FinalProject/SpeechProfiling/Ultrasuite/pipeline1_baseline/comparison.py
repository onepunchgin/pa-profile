"""Reference-vs-hypothesis comparison for English child speech.

Mirrors `pipeline1_baseline/comparison.py` shape (word/syllable/char/
phoneme error rates + a `pattern` counter struct) so the SSD scorer
plugs in unchanged.

The pattern counters here encode SSD error patterns common in English
child speech (per Bowen, Bernhardt, Stoel-Gammon clinical taxonomies):
  - cluster_reduction: 'spoon'→'poon', 'tree'→'tee'
  - final_consonant_deletion: 'cat'→'ca'
  - stopping: fricative→stop, 'see'→'tee', 'fish'→'pish'
  - fronting: velar→alveolar, 'go'→'do', 'cake'→'tate'
  - gliding: liquid→glide, 'rabbit'→'wabbit', 'lion'→'yion'
  - voicing: initial voicing/devoicing, 'bee'→'pee'
  - weak_syllable_deletion: 'banana'→'nana', 'elephant'→'ephant'
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Sequence

from .g2p import text_to_chars, text_to_phones, text_to_syllables, text_to_words


@dataclass
class ErrorRate:
    n_ref: int = 0
    n_hyp: int = 0
    substitutions: int = 0
    deletions:     int = 0
    insertions:    int = 0
    error_rate:    float = 0.0


@dataclass
class PatternCounts:
    cluster_reduction: int = 0
    final_consonant_deletion: int = 0
    stopping: int = 0
    fronting: int = 0
    gliding: int = 0
    voicing_change: int = 0
    weak_syllable_deletion: int = 0


@dataclass
class ComparisonResult:
    word: ErrorRate = field(default_factory=ErrorRate)
    syllable: ErrorRate = field(default_factory=ErrorRate)
    char: ErrorRate = field(default_factory=ErrorRate)
    phoneme: ErrorRate = field(default_factory=ErrorRate)
    pattern: PatternCounts = field(default_factory=PatternCounts)


# ── Levenshtein-style alignment + ER ─────────────────────────────────────
def _edit_align(ref: Sequence[str], hyp: Sequence[str]):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bp = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i; bp[i][0] = "D"
    for j in range(m + 1):
        dp[0][j] = j; bp[0][j] = "I"
    bp[0][0] = None
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]; bp[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1
                m_ = min(sub, deletion, insertion)
                dp[i][j] = m_
                bp[i][j] = "S" if m_ == sub else ("D" if m_ == deletion else "I")
    # Trace back
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bp[i][j]
        if op == "M":
            out.append((i - 1, j - 1, "M")); i -= 1; j -= 1
        elif op == "S":
            out.append((i - 1, j - 1, "S")); i -= 1; j -= 1
        elif op == "D":
            out.append((i - 1, None,    "D")); i -= 1
        else:
            out.append((None,    j - 1, "I")); j -= 1
    out.reverse()
    return out, dp[n][m]


def _er(ref: Sequence[str], hyp: Sequence[str]) -> ErrorRate:
    if not ref:
        return ErrorRate(n_ref=0, n_hyp=len(hyp), insertions=len(hyp),
                         error_rate=0.0 if not hyp else 1.0)
    align, dist = _edit_align(ref, hyp)
    s = sum(1 for _, _, op in align if op == "S")
    d = sum(1 for _, _, op in align if op == "D")
    i = sum(1 for _, _, op in align if op == "I")
    return ErrorRate(n_ref=len(ref), n_hyp=len(hyp),
                     substitutions=s, deletions=d, insertions=i,
                     error_rate=dist / max(len(ref), 1))


# ── English child SSD pattern counters ───────────────────────────────────
_FRICATIVES = {"S", "Z", "F", "V", "TH", "DH", "SH", "ZH"}
_STOPS      = {"P", "B", "T", "D", "K", "G"}
_VELARS     = {"K", "G", "NG"}
_ALVEOLARS  = {"T", "D", "N", "S", "Z", "L", "R"}
_LIQUIDS    = {"R", "L"}
_GLIDES     = {"W", "Y", "JH"}
_VOICED_PAIRS = {"B": "P", "D": "T", "G": "K",
                 "V": "F", "Z": "S", "DH": "TH", "JH": "CH"}


def _strip_stress(p: str) -> str:
    import re
    return re.sub(r"\d", "", p).upper()


def _count_patterns(ref_phones: List[str], hyp_phones: List[str]) -> PatternCounts:
    pc = PatternCounts()
    align, _ = _edit_align(ref_phones, hyp_phones)
    # Pass 1: per-phone substitutions classify as fronting/stopping/gliding/voicing
    for r_idx, h_idx, op in align:
        if op != "S":
            continue
        rp = _strip_stress(ref_phones[r_idx])
        hp = _strip_stress(hyp_phones[h_idx])
        if rp in _FRICATIVES and hp in _STOPS:
            pc.stopping += 1
        if rp in _VELARS and hp in _ALVEOLARS:
            pc.fronting += 1
        if rp in _LIQUIDS and hp in _GLIDES:
            pc.gliding += 1
        if rp in _VOICED_PAIRS and hp == _VOICED_PAIRS[rp]:
            pc.voicing_change += 1
        if hp in _VOICED_PAIRS and rp == _VOICED_PAIRS[hp]:
            pc.voicing_change += 1
    # Pass 2: structural — cluster reduction (consonant deleted between
    # other consonants in onset) and final-consonant deletion.
    for k, (r_idx, h_idx, op) in enumerate(align):
        if op != "D" or r_idx is None:
            continue
        rp = _strip_stress(ref_phones[r_idx])
        if rp in _FRICATIVES | _STOPS | _LIQUIDS:
            # Final consonant deletion: nothing follows in the alignment
            following = [a for a in align[k + 1:] if a[2] != "I"]
            if not following:
                pc.final_consonant_deletion += 1
            else:
                # Check if neighbour is also consonant (cluster context)
                neighbour_consonant = False
                if r_idx + 1 < len(ref_phones):
                    nxt = _strip_stress(ref_phones[r_idx + 1])
                    if nxt in _FRICATIVES | _STOPS | _LIQUIDS | _VELARS:
                        neighbour_consonant = True
                if r_idx > 0:
                    prv = _strip_stress(ref_phones[r_idx - 1])
                    if prv in _FRICATIVES | _STOPS | _LIQUIDS | _VELARS:
                        neighbour_consonant = True
                if neighbour_consonant:
                    pc.cluster_reduction += 1
    return pc


# ── Top-level entry ──────────────────────────────────────────────────────
def compare(reference_text: str, hypothesis_text: str) -> ComparisonResult:
    ref_w = text_to_words(reference_text)
    hyp_w = text_to_words(hypothesis_text)
    ref_c = text_to_chars(reference_text)
    hyp_c = text_to_chars(hypothesis_text)
    ref_p = text_to_phones(reference_text)
    hyp_p = text_to_phones(hypothesis_text)
    ref_s = [s for w in text_to_syllables(reference_text) for s in w]
    hyp_s = [s for w in text_to_syllables(hypothesis_text) for s in w]
    return ComparisonResult(
        word=_er(ref_w, hyp_w),
        char=_er(ref_c, hyp_c),
        phoneme=_er(ref_p, hyp_p),
        syllable=_er([tuple(s) for s in ref_s], [tuple(s) for s in hyp_s]),
        pattern=_count_patterns(ref_p, hyp_p),
    )
