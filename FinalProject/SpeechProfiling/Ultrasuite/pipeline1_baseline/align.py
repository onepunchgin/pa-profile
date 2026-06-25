"""English MFA aligner for UltraSuite — mirrors Kannada MFAAligner API.

Uses english_us_arpa (acoustic + dictionary) downloaded into the mfa2 env.
Produces AlignedSegment list compatible with the alignment_features()
helper from Pipeline 1 baseline (so per-phone timing flows into SSD
scoring identically).

Phone vs word note: english_us_arpa MFA outputs both Words and Phones
tiers. We use the Phones tier here (per-phone, not per-word) because
SSD child-speech features are typically phone-level (cluster reduction,
final consonant deletion, etc.).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

from FinalProject.SpeechProfiling.pipeline1_baseline.align import AlignedSegment

_MFA_BIN = os.environ.get(
    "PA_PROFILE_MFA_BIN",
    "/home/prouser1/miniconda3/envs/mfa2/bin",
)
_FAILURE_DUMP_DIR = Path(os.environ.get(
    "PA_PROFILE_MFA_FAILURE_DIR",
    "/media/csedept/lab7/FinalProject/SpeechProfiling/Ultrasuite/"
    "stage8_classifier/runs/mfa_failures",
))


def _dump_mfa_failure(tmp: Path, audio_path, reference_text: str,
                      cmd: list, exc: subprocess.CalledProcessError) -> None:
    """Preserve forensic state of a failed mfa align run before tempdir cleanup."""
    try:
        ap = Path(audio_path)
        try:
            i = ap.parts.index("core")
            tag = "_".join(ap.parts[i + 1:]).replace(".wav", "")
        except ValueError:
            tag = ap.stem
        dest = _FAILURE_DUMP_DIR / f"{int(time.time() * 1000)}_{tag}"
        dest.mkdir(parents=True, exist_ok=True)
        log_src = tmp / "mfa_tmp" / "corpus" / "corpus.log"
        if log_src.exists():
            shutil.copy(log_src, dest / "corpus.log")
        db_src = tmp / "mfa_tmp" / "corpus" / "corpus.db"
        if db_src.exists():
            existing_db_dumps = sum(1 for _ in _FAILURE_DUMP_DIR.glob("*/corpus.db"))
            if existing_db_dumps < 30:
                shutil.copy(db_src, dest / "corpus.db")
        (dest / "stderr.txt").write_bytes(exc.stderr or b"")
        (dest / "stdout.txt").write_bytes(exc.stdout or b"")
        (dest / "info.txt").write_text(
            f"audio_path={audio_path}\n"
            f"text={reference_text!r}\n"
            f"cmd={cmd}\n"
            f"exit={exc.returncode}\n"
        )
    except Exception:
        pass


class EnglishMFAAligner:
    """Kannada MFAAligner equivalent for English child speech.

    Returns AlignedSegment objects whose `char` field holds the MFA phone
    label (e.g. 'AH0', 'S', 'IH1'). Downstream alignment_features() works
    unchanged.
    """

    def __init__(self,
                 acoustic_model: str = "english_us_arpa",
                 dictionary:     str = "english_us_arpa",
                 env_bin: str = _MFA_BIN):
        self.acoustic = acoustic_model
        self.dictionary = dictionary
        self.env_bin = env_bin
        self._mfa = str(Path(env_bin) / "mfa") if env_bin else "mfa"

    def align(self, audio_path, reference_text: str) -> List[AlignedSegment]:
        from praatio import textgrid as _tg

        with tempfile.TemporaryDirectory(prefix="mfa_eng_") as tmpdir:
            tmp = Path(tmpdir)
            spk_dir = tmp / "corpus" / "spk1"
            spk_dir.mkdir(parents=True)
            shutil.copy(audio_path, spk_dir / "utt1.wav")
            (spk_dir / "utt1.lab").write_text(reference_text.strip(),
                                              encoding="utf-8")
            out_dir = tmp / "out"; out_dir.mkdir()

            env = os.environ.copy()
            if self.env_bin:
                env["PATH"] = f"{self.env_bin}:" + env.get("PATH", "")
            cmd = [
                self._mfa, "align",
                str(tmp / "corpus"), self.dictionary, self.acoustic,
                str(out_dir),
                "--temporary_directory", str(tmp / "mfa_tmp"),
                "--clean", "--num_jobs", "1",
                "--beam", "100", "--retry_beam", "400",
            ]
            try:
                subprocess.run(cmd, env=env, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as _e:
                _dump_mfa_failure(tmp, audio_path, reference_text, cmd, _e)
                raise

            tg_path = out_dir / "spk1" / "utt1.TextGrid"
            tg = _tg.openTextgrid(str(tg_path), includeEmptyIntervals=False)
            phones_tier = next(
                tg._tierDict[name] for name in tg.tierNames
                if name.lower().endswith("phones")
            )
            intervals = [(e.start, e.end, e.label) for e in phones_tier.entries
                         if e.label and e.label not in ("<eps>", "spn", "sil", "")]

        # Each MFA-emitted phone becomes one AlignedSegment.
        # `char` carries the phone label (e.g. 'AH0'); `pred_char` mirrors it.
        return [
            AlignedSegment(char=lbl, start_s=float(t0), end_s=float(t1),
                           matched=True, pred_char=lbl)
            for (t0, t1, lbl) in intervals
        ]
