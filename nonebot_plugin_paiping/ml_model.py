from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FEATURE_NAMES = [
    "rule_score",
    "frequency",
    "local_frequency",
    "chroma",
    "banding",
    "rectangle",
    "illumination",
    "softness",
    "camera_artifact",
    "display_content",
    "white_clip",
    "screen_aspect",
    "overexposed_display",
    "screenshot_penalty",
    "aspect_ratio",
    "log_area",
    "edge_density",
    "entropy",
    "gray_std",
    "dark_clip",
    "bright_clip",
    "sat_mean",
    "sat_high",
    "lap_log",
    "colorfulness",
]


@dataclass(frozen=True)
class LinearPaipingModel:
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float
    metadata: dict[str, Any]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LinearPaipingModel":
        feature_names = list(data["feature_names"])
        return cls(
            feature_names=feature_names,
            mean=np.asarray(data["mean"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            weights=np.asarray(data["weights"], dtype=np.float64),
            bias=float(data["bias"]),
            threshold=float(data["threshold"]),
            metadata={
                key: value
                for key, value in data.items()
                if key not in {"feature_names", "mean", "scale", "weights", "bias", "threshold"}
            },
        )

    def predict_probability(self, feature_values: dict[str, float]) -> float:
        vector = np.asarray(
            [float(feature_values.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        )
        scale = np.where(np.abs(self.scale) < 1e-9, 1.0, self.scale)
        logit = float(((vector - self.mean) / scale) @ self.weights + self.bias)
        return _sigmoid(logit)


def load_model(path: str | Path) -> LinearPaipingModel:
    model_path = Path(path)
    data = json.loads(model_path.read_text(encoding="utf-8"))
    return LinearPaipingModel.from_mapping(data)


def extract_model_features(image: np.ndarray, rule_result: Any) -> dict[str, float]:
    features = {name: 0.0 for name in FEATURE_NAMES}
    features["rule_score"] = float(getattr(rule_result, "score", 0.0))

    rule_features = getattr(rule_result, "features", {}) or {}
    for name, value in rule_features.items():
        if name in features:
            features[name] = float(value)

    features.update(_image_statistics(image))
    return features


def _image_statistics(image: np.ndarray) -> dict[str, float]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probability = hist / (float(hist.sum()) + 1e-9)
    nonzero = probability > 0
    entropy = float(-np.sum(probability[nonzero] * np.log2(probability[nonzero])) / 8.0)

    b_channel, g_channel, r_channel = cv2.split(image.astype(np.float32))
    red_green = np.abs(r_channel - g_channel)
    yellow_blue = np.abs(0.5 * (r_channel + g_channel) - b_channel)
    colorfulness = float(
        np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2)
        + 0.3 * np.sqrt(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
    )

    return {
        "aspect_ratio": float(max(width / float(height), height / float(width))),
        "log_area": float(np.log1p(width * height) / 16.0),
        "edge_density": edge_density,
        "entropy": entropy,
        "gray_std": float(gray.std() / 128.0),
        "dark_clip": float(np.mean(gray < 25)),
        "bright_clip": float(np.mean(gray > 245)),
        "sat_mean": float(hsv[:, :, 1].mean() / 255.0),
        "sat_high": float(np.mean(hsv[:, :, 1] > 170)),
        "lap_log": float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()) / 9.0),
        "colorfulness": float(colorfulness / 128.0),
    }


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -40.0, 40.0))
    return float(1.0 / (1.0 + np.exp(-value)))

