"""WER / CER / PER evaluation helpers."""
from __future__ import annotations

from typing import Iterable, List, Sequence

from jiwer import cer as _cer, wer as _wer


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return float(_wer(list(refs), list(hyps)))


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return float(_cer(list(refs), list(hyps)))


def edit_ops(ref: Sequence, hyp: Sequence) -> tuple[int, int, int, int]:
    """Levenshtein with op breakdown over arbitrary token sequences.

    Returns (substitutions, deletions, insertions, total_ref_len)."""
    ref = list(ref)
    hyp = list(hyp)
    n, m = len(ref), len(hyp)
    if n == 0:
        return (0, 0, m, 0)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(m + 1):
        dp[0][j] = j
        bt[0][j] = "I"
    bt[0][0] = None
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bt[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1
                best = min(sub, deletion, insertion)
                dp[i][j] = best
                bt[i][j] = "S" if best == sub else ("D" if best == deletion else "I")

    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "M":
            i -= 1; j -= 1
        elif op == "S":
            s += 1; i -= 1; j -= 1
        elif op == "D":
            d += 1; i -= 1
        elif op == "I":
            ins += 1; j -= 1
        else:
            break
    return (s, d, ins, n)


def per(ref_phones: Sequence[str], hyp_phones: Sequence[str]) -> float:
    s, d, i, n = edit_ops(ref_phones, hyp_phones)
    return (s + d + i) / max(n, 1)
