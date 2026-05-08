from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTOR_PATH = PROJECT_ROOT / "nonebot_plugin_paiping" / "detector.py"
MODEL_HELPER_PATH = PROJECT_ROOT / "nonebot_plugin_paiping" / "ml_model.py"

spec = importlib.util.spec_from_file_location("_paiping_detector", DETECTOR_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load detector from {DETECTOR_PATH}")
detector_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = detector_module
spec.loader.exec_module(detector_module)
detect_screen_photo = detector_module.detect_screen_photo

model_spec = importlib.util.spec_from_file_location("_paiping_ml_model", MODEL_HELPER_PATH)
if model_spec is None or model_spec.loader is None:
    raise RuntimeError(f"Cannot load model helper from {MODEL_HELPER_PATH}")
model_module = importlib.util.module_from_spec(model_spec)
sys.modules[model_spec.name] = model_module
model_spec.loader.exec_module(model_module)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def iter_images(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            images.extend(
                child for child in sorted(path.rglob("*")) if child.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    return images


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe screen-photo detection scores.")
    parser.add_argument("paths", nargs="+", type=Path, help="Image files or directories.")
    parser.add_argument("--threshold", type=float, default=0.62, help="Detection threshold.")
    parser.add_argument("--model", type=Path, help="Optional trained model JSON.")
    parser.add_argument("--json", action="store_true", help="Print JSON lines.")
    args = parser.parse_args()

    trained_model = model_module.load_model(args.model) if args.model else None

    for image_path in iter_images(args.paths):
        image = read_image(image_path)
        if image is None:
            print(f"SKIP {image_path} decode_failed")
            continue

        result = detect_screen_photo(image, threshold=args.threshold)
        model_probability = None
        model_detected = None
        model_threshold = None
        if trained_model is not None:
            feature_values = model_module.extract_model_features(image, result)
            model_probability = trained_model.predict_probability(feature_values)
            model_threshold = trained_model.threshold
            model_detected = model_probability >= model_threshold

        if args.json:
            payload = {
                "path": str(image_path),
                "detected": result.is_screen_photo,
                "score": result.score,
                "threshold": result.threshold,
                "features": result.features,
                "reasons": result.reasons,
            }
            if trained_model is not None:
                payload.update(
                    {
                        "model_detected": model_detected,
                        "model_probability": round(float(model_probability), 4),
                        "model_threshold": round(float(model_threshold), 4),
                    }
                )
            print(json.dumps(payload, ensure_ascii=False))
            continue

        if trained_model is not None:
            label = "PAIPING" if model_detected else "NORMAL"
            print(
                f"{label:7} model={model_probability:.3f}/{model_threshold:.3f} "
                f"rule={result.score:.3f}/{result.threshold:.3f} {image_path}"
            )
        else:
            label = "PAIPING" if result.is_screen_photo else "NORMAL"
            print(f"{label:7} score={result.score:.3f}/{result.threshold:.3f} {image_path}")
        features = " ".join(f"{key}={value:.2f}" for key, value in result.features.items())
        print(f"        {features}")
        if result.reasons:
            print(f"        {'; '.join(result.reasons)}")


if __name__ == "__main__":
    main()
