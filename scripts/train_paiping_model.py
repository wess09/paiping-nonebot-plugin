from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTOR_PATH = PROJECT_ROOT / "nonebot_plugin_paiping" / "detector.py"
MODEL_HELPER_PATH = PROJECT_ROOT / "nonebot_plugin_paiping" / "ml_model.py"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CACHE_VERSION = 1


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


detector = load_module("_paiping_detector", DETECTOR_PATH)
ml_model = load_module("_paiping_ml_model", MODEL_HELPER_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small binary paiping classifier.")
    parser.add_argument("--dataset", type=Path, default=Path("验证"), help="Dataset root.")
    parser.add_argument("--positive", default="拍屏", help="Positive class folder name.")
    parser.add_argument("--negative", default="正常", help="Negative class folder name.")
    parser.add_argument("--output", type=Path, default=Path("data/paiping/model.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--epochs", type=int, default=3600)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.035)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for image feature extraction. Use 0 for all CPU cores.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="JSONL feature cache. Cached entries are reused when path, size and mtime match.",
    )
    args = parser.parse_args()

    workers = resolve_workers(args.workers)
    samples = load_samples(
        args.dataset,
        args.positive,
        args.negative,
        workers=workers,
        cache_path=args.cache,
    )
    if len(samples) < 20:
        raise SystemExit("Need at least 20 images to train a useful model.")

    x = np.vstack([sample["features"] for sample in samples])
    y = np.asarray([sample["label"] for sample in samples], dtype=np.float64)
    paths = [sample["path"] for sample in samples]

    rule_metrics = compute_metrics((x[:, 0] >= 0.62).astype(int), y)
    folds = make_stratified_folds(y, args.folds, args.seed)
    oof_probability = np.zeros(len(y), dtype=np.float64)
    fold_metrics = []

    for index, (train_indices, test_indices) in enumerate(folds, start=1):
        model = fit_logistic_regression(
            x[train_indices],
            y[train_indices],
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        train_probability = predict_probability(x[train_indices], model)
        threshold = best_threshold(train_probability, y[train_indices])
        test_probability = predict_probability(x[test_indices], model)
        oof_probability[test_indices] = test_probability
        prediction = (test_probability >= threshold).astype(int)
        metrics = compute_metrics(prediction, y[test_indices])
        metrics["threshold"] = round(float(threshold), 4)
        metrics["fold"] = index
        fold_metrics.append(metrics)

    oof_threshold = best_threshold(oof_probability, y)
    oof_prediction = (oof_probability >= oof_threshold).astype(int)
    oof_metrics = compute_metrics(oof_prediction, y)
    oof_metrics["threshold"] = round(float(oof_threshold), 4)

    final_model = fit_logistic_regression(
        x,
        y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    final_probability = predict_probability(x, final_model)
    train_prediction = (final_probability >= oof_threshold).astype(int)
    train_metrics = compute_metrics(train_prediction, y)
    train_metrics["threshold"] = round(float(oof_threshold), 4)

    false_positive_paths = [
        str(paths[index]) for index in np.where((oof_prediction == 1) & (y == 0))[0]
    ]
    false_negative_paths = [
        str(paths[index]) for index in np.where((oof_prediction == 0) & (y == 1))[0]
    ]

    save_model(
        args.output,
        final_model,
        threshold=oof_threshold,
        dataset=args.dataset,
        positive=args.positive,
        negative=args.negative,
        positive_count=int(y.sum()),
        negative_count=int((1.0 - y).sum()),
        rule_metrics=rule_metrics,
        fold_metrics=fold_metrics,
        oof_metrics=oof_metrics,
        train_metrics=train_metrics,
        false_positive_paths=false_positive_paths,
        false_negative_paths=false_negative_paths,
    )

    print_summary(
        output=args.output,
        count=len(y),
        positive_count=int(y.sum()),
        negative_count=int((1.0 - y).sum()),
        rule_metrics=rule_metrics,
        fold_metrics=fold_metrics,
        oof_metrics=oof_metrics,
        train_metrics=train_metrics,
        false_positive_paths=false_positive_paths,
        false_negative_paths=false_negative_paths,
    )


def resolve_workers(value: int) -> int:
    if value <= 0:
        return max(1, os.cpu_count() or 1)
    return max(1, int(value))


def load_samples(
    dataset: Path,
    positive: str,
    negative: str,
    *,
    workers: int = 1,
    cache_path: Path | None = None,
) -> list[dict[str, object]]:
    items = collect_image_items(dataset, positive, negative)
    cache = load_feature_cache(cache_path)
    sample_by_key: dict[str, dict[str, object]] = {}
    cache_entries: dict[str, dict[str, object]] = {}
    missing_items = []

    for item in items:
        entry = cache.get(item["cache_key"])
        if is_cache_entry_valid(entry, item):
            sample_by_key[item["cache_key"]] = sample_from_cache_entry(item, entry)
            cache_entries[item["cache_key"]] = entry
        else:
            missing_items.append(item)

    if cache_path is not None or workers > 1:
        print(
            "feature_cache="
            f"{cache_path if cache_path is not None else '-'} "
            f"hit={len(items) - len(missing_items)} miss={len(missing_items)} "
            f"workers={workers}"
        )

    if missing_items:
        item_by_key = {item["cache_key"]: item for item in missing_items}
        for result in extract_feature_results(missing_items, workers):
            key = str(result.get("path", ""))
            item = item_by_key.get(key)
            if item is None:
                continue
            if not result.get("ok"):
                print(f"SKIP {result.get('error', 'extract_failed')} {item['path']}")
                continue
            features = np.asarray(result["features"], dtype=np.float64)
            sample_by_key[key] = {
                "path": item["path"],
                "label": item["label"],
                "features": features,
            }
            cache_entries[key] = make_cache_entry(item, features)

    if cache_path is not None:
        ordered_entries = [
            cache_entries[item["cache_key"]]
            for item in items
            if item["cache_key"] in cache_entries
        ]
        write_feature_cache(cache_path, ordered_entries)
        print(f"cache_written={cache_path} entries={len(ordered_entries)}")

    return [
        sample_by_key[item["cache_key"]]
        for item in items
        if item["cache_key"] in sample_by_key
    ]


def collect_image_items(dataset: Path, positive: str, negative: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for folder, label in ((positive, 1), (negative, 0)):
        root = dataset / folder
        if not root.exists():
            raise SystemExit(f"Dataset folder does not exist: {root}")
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                stat = path.stat()
            except OSError as exc:
                print(f"SKIP stat_failed {path}: {exc}")
                continue
            items.append(
                {
                    "path": path,
                    "label": label,
                    "cache_key": str(path.resolve()),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "size": int(stat.st_size),
                }
            )
    return items


def load_feature_cache(cache_path: Path | None) -> dict[str, dict[str, object]]:
    if cache_path is None or not cache_path.exists():
        return {}

    cache: dict[str, dict[str, object]] = {}
    with cache_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"SKIP cache_bad_json line={line_number} {cache_path}")
                continue
            key = entry.get("path")
            if isinstance(key, str):
                cache[key] = entry
    return cache


def is_cache_entry_valid(entry: dict[str, object] | None, item: dict[str, object]) -> bool:
    if not entry:
        return False
    if entry.get("cache_version") != CACHE_VERSION:
        return False
    if entry.get("label") != item["label"]:
        return False
    if entry.get("mtime_ns") != item["mtime_ns"] or entry.get("size") != item["size"]:
        return False
    if entry.get("feature_names") != ml_model.FEATURE_NAMES:
        return False
    features = entry.get("features")
    return isinstance(features, list) and len(features) == len(ml_model.FEATURE_NAMES)


def sample_from_cache_entry(
    item: dict[str, object],
    entry: dict[str, object],
) -> dict[str, object]:
    return {
        "path": item["path"],
        "label": item["label"],
        "features": np.asarray(entry["features"], dtype=np.float64),
    }


def extract_feature_results(
    items: list[dict[str, object]],
    workers: int,
) -> list[dict[str, object]]:
    tasks = [(str(item["cache_key"]), int(item["label"])) for item in items]
    if workers <= 1 or len(tasks) <= 1:
        return [extract_feature_worker(task) for task in tasks]

    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        future_map = {executor.submit(extract_feature_worker, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "ok": False,
                        "path": task[0],
                        "label": task[1],
                        "error": f"worker_failed: {type(exc).__name__}: {exc}",
                    }
                )
    return results


def extract_feature_worker(task: tuple[str, int]) -> dict[str, object]:
    path_text, label = task
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    try:
        image = read_image(Path(path_text))
        if image is None:
            return {"ok": False, "path": path_text, "label": label, "error": "decode_failed"}
        rule_result = detector.detect_screen_photo(image)
        features = ml_model.extract_model_features(image, rule_result)
        return {
            "ok": True,
            "path": path_text,
            "label": label,
            "features": [float(features[name]) for name in ml_model.FEATURE_NAMES],
        }
    except Exception as exc:
        return {
            "ok": False,
            "path": path_text,
            "label": label,
            "error": f"{type(exc).__name__}: {exc}",
        }


def make_cache_entry(item: dict[str, object], features: np.ndarray) -> dict[str, object]:
    return {
        "cache_version": CACHE_VERSION,
        "path": item["cache_key"],
        "relative_path": str(item["path"]),
        "label": int(item["label"]),
        "mtime_ns": int(item["mtime_ns"]),
        "size": int(item["size"]),
        "feature_names": list(ml_model.FEATURE_NAMES),
        "features": [float(value) for value in np.asarray(features).ravel()],
    }


def write_feature_cache(cache_path: Path, entries: list[dict[str, object]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
    tmp_path.replace(cache_path)


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def make_stratified_folds(y: np.ndarray, folds: int, seed: int):
    folds = max(2, min(int(folds), int(np.bincount(y.astype(int)).min())))
    rng = random.Random(seed)
    positive = [index for index, value in enumerate(y) if value == 1]
    negative = [index for index, value in enumerate(y) if value == 0]
    rng.shuffle(positive)
    rng.shuffle(negative)

    result = []
    all_indices = set(range(len(y)))
    for fold_index in range(folds):
        test_indices = positive[fold_index::folds] + negative[fold_index::folds]
        test_set = set(test_indices)
        train_indices = sorted(all_indices - test_set)
        result.append((np.asarray(train_indices), np.asarray(test_indices)))
    return result


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> dict[str, np.ndarray | float]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale

    count, feature_count = z.shape
    weights = np.zeros(feature_count, dtype=np.float64)
    bias = 0.0
    positive_count = float(np.sum(y == 1))
    negative_count = float(np.sum(y == 0))
    sample_weight = np.where(y == 1, count / (2.0 * positive_count), count / (2.0 * negative_count))

    lr = learning_rate
    first_decay = int(epochs * 0.39)
    second_decay = int(epochs * 0.72)
    for epoch in range(epochs):
        probability = sigmoid(z @ weights + bias)
        error = (probability - y) * sample_weight
        weights -= lr * ((z.T @ error) / count + l2 * weights)
        bias -= lr * float(error.mean())
        if epoch in {first_decay, second_decay}:
            lr *= 0.45

    return {"mean": mean, "scale": scale, "weights": weights, "bias": bias}


def predict_probability(x: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    mean = model["mean"]
    scale = model["scale"]
    weights = model["weights"]
    bias = float(model["bias"])
    return sigmoid(((x - mean) / scale) @ weights + bias)


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def best_threshold(probability: np.ndarray, y: np.ndarray) -> float:
    best_f1 = -1.0
    best = 0.5
    for threshold in np.linspace(0.15, 0.85, 141):
        metrics = compute_metrics((probability >= threshold).astype(int), y)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best = float(threshold)
    return best


def compute_metrics(prediction: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((prediction == 1) & (y == 1)))
    tn = int(np.sum((prediction == 0) & (y == 0)))
    fp = int(np.sum((prediction == 1) & (y == 0)))
    fn = int(np.sum((prediction == 0) & (y == 1)))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / max(1, len(y))
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def save_model(
    output: Path,
    model: dict[str, np.ndarray | float],
    *,
    threshold: float,
    dataset: Path,
    positive: str,
    negative: str,
    positive_count: int,
    negative_count: int,
    rule_metrics: dict[str, float | int],
    fold_metrics: list[dict[str, float | int]],
    oof_metrics: dict[str, float | int],
    train_metrics: dict[str, float | int],
    false_positive_paths: list[str],
    false_negative_paths: list[str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "nonebot-plugin-paiping-linear-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": ml_model.FEATURE_NAMES,
        "mean": as_list(model["mean"]),
        "scale": as_list(model["scale"]),
        "weights": as_list(model["weights"]),
        "bias": round(float(model["bias"]), 10),
        "threshold": round(float(threshold), 6),
        "dataset": {
            "root": str(dataset),
            "positive": positive,
            "negative": negative,
            "positive_count": positive_count,
            "negative_count": negative_count,
        },
        "metrics": {
            "rule_baseline": rule_metrics,
            "folds": fold_metrics,
            "out_of_fold": oof_metrics,
            "train_all": train_metrics,
            "false_positive_paths": false_positive_paths,
            "false_negative_paths": false_negative_paths,
        },
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def as_list(value) -> list[float]:
    return [round(float(item), 10) for item in np.asarray(value).ravel()]


def print_summary(
    *,
    output: Path,
    count: int,
    positive_count: int,
    negative_count: int,
    rule_metrics: dict[str, float | int],
    fold_metrics: list[dict[str, float | int]],
    oof_metrics: dict[str, float | int],
    train_metrics: dict[str, float | int],
    false_positive_paths: list[str],
    false_negative_paths: list[str],
) -> None:
    print(f"samples={count} positive={positive_count} negative={negative_count}")
    print(f"rule_baseline={format_metrics(rule_metrics)}")
    for metrics in fold_metrics:
        print(f"fold{metrics['fold']}={format_metrics(metrics)} threshold={metrics['threshold']}")
    print(f"out_of_fold={format_metrics(oof_metrics)} threshold={oof_metrics['threshold']}")
    print(f"train_all={format_metrics(train_metrics)} threshold={train_metrics['threshold']}")
    print(f"false_positive={len(false_positive_paths)} false_negative={len(false_negative_paths)}")
    print(f"saved={output}")


def format_metrics(metrics: dict[str, float | int]) -> str:
    return (
        f"acc={metrics['accuracy']} precision={metrics['precision']} "
        f"recall={metrics['recall']} f1={metrics['f1']} "
        f"tp={metrics['tp']} tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']}"
    )


if __name__ == "__main__":
    main()
