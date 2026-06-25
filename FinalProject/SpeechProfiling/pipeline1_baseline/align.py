"""Forced alignment between reference text and audio.

Strategy: CTC-greedy alignment using the data2vec_kn ASR logits is the
default — fast, no extra deps. `MFAAligner` provides a higher-quality
alternative that uses a trained Kannada MFA acoustic model
(`shared/mfa_kannada/runs/v1/kannada_v2b.zip`).

For each reference character (after Kannada-only normalization) we return:
  {'char', 'start_s', 'end_s', 'matched': bool, 'pred_char': str | None}

CTC alignment approach:
  1. Run ASR forward → logits (T, V).
  2. Greedy argmax → per-frame token id.
  3. Walk forward, collapsing repeats and blanks; record (token_id, t_start,
     t_end) for every emitted token.
  4. Compute Levenshtein alignment between the emitted-token sequence and
     reference-character sequence; per ref char that maps to an emitted
     token, take that token's time bounds. Unmatched ref chars get None.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from FinalProject.shared.audio_io import TARGET_SR, load_audio
from FinalProject.shared.text_norm import KANNADA_RANGE, normalize

from .asr import ASRModel

# CTC encoder hop (data2vec / wav2vec2 standard): 320 samples per frame at 16k = 20ms
CTC_FRAME_HOP_S = 320.0 / TARGET_SR


@dataclass
class AlignedSegment:
    char: str
    start_s: Optional[float]
    end_s: Optional[float]
    matched: bool
    pred_char: Optional[str]


def alignment_features(segments: List["AlignedSegment"]) -> Dict[str, float]:
    """Per-utterance timing stats derived from forced alignment.

    Backend-agnostic: same shape for CTC and MFA outputs. Useful as input
    to SSD heuristics that care about per-phone timing irregularity (CAS,
    dysarthria) and intra-utterance pauses (fluency).
    """
    timed = [s for s in segments if s.start_s is not None and s.end_s is not None]
    out = {
        "align_n_chars_aligned": float(len(timed)),
        "align_n_chars_unaligned": float(len(segments) - len(timed)),
        "align_char_dur_mean_s": 0.0,
        "align_char_dur_std_s": 0.0,
        "align_char_dur_cv": 0.0,
        "align_char_dur_max_s": 0.0,
        "align_articulation_rate_cps": 0.0,
        "align_intra_pause_total_s": 0.0,
        "align_intra_pause_max_s": 0.0,
        "align_intra_pause_count": 0.0,
    }
    if not timed:
        return out

    durs = [max(0.0, s.end_s - s.start_s) for s in timed]
    n = len(durs)
    mean = sum(durs) / n
    var = sum((d - mean) ** 2 for d in durs) / n
    std = var ** 0.5
    out["align_char_dur_mean_s"] = mean
    out["align_char_dur_std_s"] = std
    out["align_char_dur_cv"] = std / mean if mean > 0 else 0.0
    out["align_char_dur_max_s"] = max(durs)

    span = timed[-1].end_s - timed[0].start_s
    out["align_articulation_rate_cps"] = n / span if span > 0 else 0.0

    # Intra-utterance silences = gaps between consecutive aligned chars.
    gaps = []
    for prev, curr in zip(timed, timed[1:]):
        g = curr.start_s - prev.end_s
        if g > 0.05:  # 50 ms threshold (below this is just micro-jitter)
            gaps.append(g)
    if gaps:
        out["align_intra_pause_total_s"] = sum(gaps)
        out["align_intra_pause_max_s"] = max(gaps)
        out["align_intra_pause_count"] = float(len(gaps))
    return out


def _greedy_emissions(logits: torch.Tensor, vocab: Dict[int, str]) -> List[Tuple[str, int, int]]:
    """Collapse CTC argmax stream into (token, t_start_frame, t_end_frame).

    Treats id 0 as blank. Word-boundary token '|' is dropped from the emitted
    sequence (we align over Kannada chars, spaces handled separately).
    """
    pred = logits.argmax(dim=-1).cpu().tolist()
    emissions: List[Tuple[str, int, int]] = []
    prev = -1
    seg_start = 0
    for t, tok in enumerate(pred):
        if tok == prev:
            continue
        # close previous non-blank segment
        if prev not in (-1, 0):
            ch = vocab.get(prev, "")
            if ch and ch != "|":
                emissions.append((ch, seg_start, t))
        seg_start = t
        prev = tok
    # close trailing
    if prev not in (-1, 0):
        ch = vocab.get(prev, "")
        if ch and ch != "|":
            emissions.append((ch, seg_start, len(pred)))
    return emissions


def _levenshtein_align(ref: List[str], hyp: List[str]) -> List[Tuple[int, Optional[int], str]]:
    """Return list of (ref_idx, hyp_idx_or_None, op) where op ∈ {M,S,D}.

    Insertions in hyp (no ref counterpart) are NOT returned — we only emit
    per-ref-position decisions so callers can iterate ref-char-by-ref-char.
    """
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
                dp[i][j] = 1 + min(dp[i - 1][j - 1],
                                   dp[i - 1][j],
                                   dp[i][j - 1])
    out: List[Tuple[int, Optional[int], str]] = []
    i, j = n, m
    while i > 0:
        if j > 0 and ref[i - 1] == hyp[j - 1]:
            out.append((i - 1, j - 1, "M")); i -= 1; j -= 1
        else:
            sub = dp[i - 1][j - 1] if j > 0 else 10**9
            deletion = dp[i - 1][j]
            insertion = dp[i][j - 1] if j > 0 else 10**9
            best = min(sub, deletion, insertion)
            if best == sub and j > 0:
                out.append((i - 1, j - 1, "S")); i -= 1; j -= 1
            elif best == deletion:
                out.append((i - 1, None, "D")); i -= 1
            else:
                j -= 1
    out.reverse()
    return out


def ctc_align(asr: ASRModel, audio_path, reference_text: str) -> List[AlignedSegment]:
    waveform, _ = load_audio(audio_path, target_sr=TARGET_SR)
    logits = asr.logits(waveform)
    emissions = _greedy_emissions(logits, asr.vocab)
    hyp_chars = [tok for (tok, _, _) in emissions]

    # Reference: Kannada chars only, no whitespace
    ref_chars = [ch for ch in normalize(reference_text)
                 if KANNADA_RANGE[0] <= ord(ch) <= KANNADA_RANGE[1]]

    decisions = _levenshtein_align(ref_chars, hyp_chars)

    segments: List[AlignedSegment] = []
    for ref_idx, hyp_idx, op in decisions:
        ref_ch = ref_chars[ref_idx]
        if hyp_idx is not None:
            ch, t0, t1 = emissions[hyp_idx]
            segments.append(AlignedSegment(
                char=ref_ch,
                start_s=t0 * CTC_FRAME_HOP_S,
                end_s=t1 * CTC_FRAME_HOP_S,
                matched=(op == "M"),
                pred_char=ch if op != "D" else None,
            ))
        else:
            segments.append(AlignedSegment(
                char=ref_ch, start_s=None, end_s=None,
                matched=False, pred_char=None,
            ))
    return segments


_MFA_BIN = os.environ.get(
    "PA_PROFILE_MFA_BIN",
    "/home/prouser1/miniconda3/envs/mfa2/bin",
)
_DEFAULT_MFA_MODEL = Path(os.environ.get(
    "PA_PROFILE_KANNADA_MFA_ZIP",
    "/media/csedept/lab7/FinalProject/shared/mfa_kannada/runs/v1/kannada_v2b.zip",
))
_DEFAULT_MFA_DICT = Path(os.environ.get(
    "PA_PROFILE_KANNADA_PRON_DICT",
    "/media/csedept/lab7/FinalProject/shared/mfa_kannada/runs/v1/pron_dict.txt",
))


class MFAAligner:
    """Forced-alignment via the Kannada MFA acoustic model (kannada_v2b.zip).

    The model treats each Kannada character as a single phone (char-as-phone
    pron dict). The reference text is split into chars; each char becomes
    one MFA "word" so the Words tier of the output TextGrid maps 1:1 to
    ref-char positions (modulo dict-OOV chars which are skipped).

    Interface mirrors `ctc_align` so the pipeline orchestrator can swap
    backends without code changes.
    """

    def __init__(
        self,
        model_zip_path: os.PathLike = _DEFAULT_MFA_MODEL,
        pron_dict_path: os.PathLike = _DEFAULT_MFA_DICT,
        env_bin: str = _MFA_BIN,
    ):
        self.model_zip = str(model_zip_path)
        self.pron_dict = str(pron_dict_path)
        self.env_bin = env_bin
        self._mfa = str(Path(env_bin) / "mfa") if env_bin else "mfa"
        self._known_chars = self._load_dict_vocab(self.pron_dict)

    @staticmethod
    def _load_dict_vocab(path: str) -> set:
        vocab = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    vocab.add(parts[0])
        return vocab

    def align(self, audio_path, reference_text: str) -> List[AlignedSegment]:
        # praatio is only needed when this method runs.
        from praatio import textgrid as _tg

        ref_chars_full = [ch for ch in normalize(reference_text)
                          if KANNADA_RANGE[0] <= ord(ch) <= KANNADA_RANGE[1]]

        with tempfile.TemporaryDirectory(prefix="mfa_align_") as tmpdir:
            tmp = Path(tmpdir)
            spk_dir = tmp / "corpus" / "spk1"
            spk_dir.mkdir(parents=True)
            shutil.copy(audio_path, spk_dir / "utt1.wav")
            (spk_dir / "utt1.lab").write_text(
                " ".join(ch for ch in ref_chars_full if ch in self._known_chars),
                encoding="utf-8",
            )
            out_dir = tmp / "out"; out_dir.mkdir()

            env = os.environ.copy()
            if self.env_bin:
                env["PATH"] = f"{self.env_bin}:" + env.get("PATH", "")
            cmd = [
                self._mfa, "align",
                str(tmp / "corpus"), self.pron_dict, self.model_zip, str(out_dir),
                "--temporary_directory", str(tmp / "mfa_tmp"),
                "--clean", "--num_jobs", "1",
                "--beam", "100", "--retry_beam", "400",
            ]
            subprocess.run(cmd, env=env, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            tg_path = out_dir / "spk1" / "utt1.TextGrid"
            tg = _tg.openTextgrid(str(tg_path), includeEmptyIntervals=False)
            words_tier = next(
                tg._tierDict[name] for name in tg.tierNames
                if name.lower().endswith("words")
            )
            intervals = [(e.start, e.end, e.label) for e in words_tier.entries
                         if e.label and e.label != "<eps>"]

        # Iterate ref chars, consume intervals only for in-dict chars.
        segments: List[AlignedSegment] = []
        ii = 0
        for ch in ref_chars_full:
            if ch in self._known_chars and ii < len(intervals):
                t0, t1, lbl = intervals[ii]; ii += 1
                segments.append(AlignedSegment(
                    char=ch, start_s=float(t0), end_s=float(t1),
                    matched=(lbl == ch), pred_char=lbl,
                ))
            else:
                segments.append(AlignedSegment(
                    char=ch, start_s=None, end_s=None,
                    matched=False, pred_char=None,
                ))
        return segments
