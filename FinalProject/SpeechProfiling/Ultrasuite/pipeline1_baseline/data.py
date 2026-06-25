"""Discover UXSSD (audio, text, speaker, session) utterances.

UltraSuite UXSSD layout:
  core-uxssd/
    core/
      <speaker>/<session>/<NNN>A.{wav, txt, ult, param}

We use only `.wav` + `.txt`. The `.txt` file format (verified):
    <prompt text>
    <date> <time>
    <speaker>-<session>,

So the prompt text is line 1 (everything before the first newline).

Sessions:
  BL[12] = baseline pre-therapy (most disordered)
  Mid    = halfway through therapy
  Post   = immediately after therapy
  Maint[12] = maintenance some time after therapy
  Therapy_NN = therapy session N
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

UXSSD_ROOT = Path("/home/prouser1/core-uxssd/core")
UXSSD_LEX  = Path("/home/prouser1/core-uxssd/doc/uxssd.lex")
UXTD_ROOT  = Path("/media/csedept/lab7/core-uxtd/core")
UXTD_LEX   = Path("/media/csedept/lab7/core-uxtd/doc/uxtd.lex")  # may not exist; fallback to uxssd.lex


@dataclass(frozen=True)
class UxssdUtt:
    speaker: str          # e.g. "01M"
    session: str          # e.g. "BL1", "Therapy_03", "Post"
    utt_id:  str          # e.g. "001A"
    wav:     Path
    text:    str          # prompt only, stripped


_SESSION_KIND_RE = re.compile(r"^(BL\d*|Mid|Post|Maint\d*|Therapy_\d+)$")


def session_kind(session: str) -> str:
    """Coarse category: 'baseline' / 'therapy' / 'post' / 'maintenance' / 'mid' / 'unknown'."""
    if session.startswith("BL"):       return "baseline"
    if session == "Mid":               return "mid"
    if session == "Post":              return "post"
    if session.startswith("Maint"):    return "maintenance"
    if session.startswith("Therapy_"): return "therapy"
    return "unknown"


def _read_prompt(txt_path: Path) -> Optional[str]:
    try:
        with open(txt_path, encoding="utf-8") as f:
            first = f.readline().strip()
        return first or None
    except Exception:
        return None


def discover_uxssd(speakers: Optional[List[str]] = None,
                   sessions: Optional[List[str]] = None,
                   limit:    Optional[int] = None) -> List[UxssdUtt]:
    """Yield UxssdUtt records.

    speakers: subset (e.g. ['01M', '03F']); None = all 8.
    sessions: subset (e.g. ['BL1', 'BL2']); None = all sessions.
    limit:    cap on returned count (post-shuffle this is up to caller).
    """
    out: List[UxssdUtt] = []
    spk_dirs = sorted(p for p in UXSSD_ROOT.iterdir() if p.is_dir())
    if speakers:
        wanted = set(speakers)
        spk_dirs = [d for d in spk_dirs if d.name in wanted]
    for spk_dir in spk_dirs:
        ses_dirs = sorted(p for p in spk_dir.iterdir() if p.is_dir())
        if sessions:
            wanted = set(sessions)
            ses_dirs = [d for d in ses_dirs if d.name in wanted]
        for ses_dir in ses_dirs:
            for wav in sorted(ses_dir.glob("*.wav")):
                txt = wav.with_suffix(".txt")
                if not txt.exists():
                    continue
                prompt = _read_prompt(txt)
                if not prompt:
                    continue
                out.append(UxssdUtt(
                    speaker=spk_dir.name,
                    session=ses_dir.name,
                    utt_id=wav.stem,
                    wav=wav,
                    text=prompt,
                ))
                if limit and len(out) >= limit:
                    return out
    return out


def load_lexicon() -> dict[str, list[str]]:
    """uxssd.lex format: '<WORD> <phn1> <phn2> ...' — return {WORD: [phones]}."""
    lex: dict[str, list[str]] = {}
    with open(UXSSD_LEX, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            lex[parts[0].upper()] = parts[1:]
    return lex


def discover_uxtd(speakers: Optional[List[str]] = None,
                  limit: Optional[int] = None) -> List[UxssdUtt]:
    """Yield UxssdUtt records from the UXTD typically-developing corpus.

    UXTD layout is flatter than UXSSD — `core/<speaker>/<NNNL>.{wav,txt}`
    with no per-session subdirectory. We synthesize a `session='UXTD'`
    label so downstream code can disambiguate corpus origin uniformly.
    """
    out: List[UxssdUtt] = []
    spk_dirs = sorted(p for p in UXTD_ROOT.iterdir() if p.is_dir())
    if speakers:
        wanted = set(speakers)
        spk_dirs = [d for d in spk_dirs if d.name in wanted]
    for spk_dir in spk_dirs:
        for wav in sorted(spk_dir.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if not txt.exists():
                continue
            prompt = _read_prompt(txt)
            if not prompt:
                continue
            out.append(UxssdUtt(
                speaker=spk_dir.name,
                session="UXTD",
                utt_id=wav.stem,
                wav=wav,
                text=prompt,
            ))
            if limit and len(out) >= limit:
                return out
    return out


if __name__ == "__main__":
    # Quick smoke test
    utts = discover_uxssd(limit=20)
    print(f"discovered {len(utts)} utts (limit 20)")
    for u in utts[:5]:
        print(f"  {u.speaker}/{u.session}/{u.utt_id}  text={u.text!r}")
    lex = load_lexicon()
    print(f"lexicon: {len(lex)} words")
    for w in list(lex)[:5]:
        print(f"  {w} -> {' '.join(lex[w])}")
