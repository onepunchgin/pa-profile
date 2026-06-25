"""Stage 8 — learned SSD-vs-TD predictor (drop-in replacement for the
rule-based scorer).

Loads `runs/classifier_v1/model.joblib` (trained by `train_classifier.py`),
exposes a `predict(acoustic, align_features) -> dict` whose return shape
mirrors the rule-based `score()` output so the pipeline orchestrator can
swap between rule-based and learned heads via a flag.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

DEFAULT_MODEL = Path(os.environ.get(
    "PA_PROFILE_STAGE8_MODEL",
    "/media/csedept/lab7/FinalProject/SpeechProfiling/Ultrasuite/"
    "stage8_classifier/runs/classifier_v1/model.joblib",
))


class LearnedSSDClassifier:
    """Wraps the joblib bundle (scaler + LR + MLP + feature_keys).

    The bundle was trained as a binary SSD-vs-TD classifier on speaker-
    disjoint UXSSD/UXTD data using the 27-feature acoustic + alignment
    vector. We keep the same vector at inference time.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL):
        bundle = joblib.load(model_path)
        self.scaler = bundle["scaler"]
        self.lr     = bundle["lr"]
        self.mlp    = bundle["mlp"]
        self.feature_keys = bundle["feature_keys"]
        self.model_path = str(model_path)

    def _vector(self, acoustic: Dict[str, float],
                align_features: Optional[Dict[str, float]]) -> np.ndarray:
        af = align_features or {}
        row = []
        for k in self.feature_keys:
            v = acoustic.get(k, None)
            if v is None:
                v = af.get(k, 0.0)
            row.append(float(v or 0.0))
        return np.array(row, dtype=np.float32).reshape(1, -1)

    def predict(self,
                acoustic: Dict[str, float],
                align_features: Optional[Dict[str, float]] = None,
                model: str = "mlp") -> Dict[str, float]:
        """Return Pipeline-1-shaped probabilities + provenance.

        model: 'mlp' (default, AUC 0.78) or 'lr' (AUC 0.64).

        Returned dict shape mirrors the rule-based scorer's
        `binary_normal_vs_ssd` block plus a flat per-class block:
            {
              "ssd_prob_pct":    74.2,    # P(SSD) × 100
              "normal_prob_pct": 25.8,    # P(TD)  × 100
              "model":           "mlp",
              "model_path":      "..../classifier_v1/model.joblib",
              "feature_vector":  [...],
            }
        """
        x = self._vector(acoustic, align_features)
        x_s = self.scaler.transform(x)
        clf = self.mlp if model == "mlp" else self.lr
        prob_ssd = float(clf.predict_proba(x_s)[0, 1])
        return {
            "ssd_prob_pct":    prob_ssd * 100.0,
            "normal_prob_pct": (1.0 - prob_ssd) * 100.0,
            "model":           model,
            "model_path":      self.model_path,
            "feature_vector":  x[0].tolist(),
        }
