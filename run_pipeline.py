#!/usr/bin/env python3
"""Train the LSB detector on the local ALASKA v2 cover dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from classifier import Classifier
from dataset_pipeline import DatasetPipeline
from generate_dataset import DEFAULT_ALASKA_DIR, DEFAULT_TARGET_SIZE, prepare_alaska_lsb_dataset


def run_full_pipeline(
    alaska_dir: str = DEFAULT_ALASKA_DIR,
    output_dir: str = "data",
    n_images: int | None = None,
    img_size: int = DEFAULT_TARGET_SIZE,
    tune: bool = False,
    workers: int = -1,
    force_stego: bool = False,
    force_features: bool = False,
) -> dict:
    started = time.time()
    print("=" * 72)
    print("ALASKA v2 LSB steganography detector training")
    print("=" * 72)
    print(f"ALASKA dir : {alaska_dir}")
    print(f"image size : {img_size}")
    print(f"workers    : {workers}")
    print(f"sample     : {'all covers' if not n_images else n_images}")

    labels_path = prepare_alaska_lsb_dataset(
        alaska_dir=alaska_dir,
        output_dir=output_dir,
        n_images=n_images,
        target_size=img_size,
        workers=workers if workers and workers > 0 else None,
        force=force_stego,
    )

    processed_dir = str(Path(output_dir) / "processed")
    pipeline = DatasetPipeline(labels_path, img_size=img_size, workers=workers)
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.build_features(
        output_dir=processed_dir,
        force=force_features,
    )

    print(f"Training matrix: {X_train.shape}, validation matrix: {X_val.shape}, test matrix: {X_test.shape}")

    clf = Classifier(workers=workers)
    clf.train(X_train, y_train, tune=tune)
    print("\nCalibrating decision thresholds on validation split")
    clf.calibrate_thresholds(X_val, y_val)
    metrics = clf.evaluate(X_test, y_test, results_dir="results")
    clf.save_models("models")

    summary = {
        "alaska_dir": alaska_dir,
        "img_size": img_size,
        "n_cover_images": int((len(y_train) + len(y_val) + len(y_test)) // 2),
        "features": int(X_train.shape[1]),
        "train_images": int(len(y_train)),
        "val_images": int(len(y_val)),
        "test_images": int(len(y_test)),
        "seconds": round(time.time() - started, 2),
        "metrics": metrics,
    }
    Path("results").mkdir(exist_ok=True)
    with Path("results/training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone")
    print(f"Total time: {summary['seconds']:.1f}s")
    print("Models: models/svm_model.pkl, models/rf_model.pkl")
    print("Metrics: results/metrics.json")
    return summary


def run_train_only(tune: bool = False, workers: int = -1) -> dict:
    processed = Path("data/processed")
    X_train = np.load(processed / "features_train.npy")
    X_val = np.load(processed / "features_val.npy")
    X_test = np.load(processed / "features_test.npy")
    y_train = np.load(processed / "y_train.npy")
    y_val = np.load(processed / "y_val.npy")
    y_test = np.load(processed / "y_test.npy")

    clf = Classifier(workers=workers)
    clf.train(X_train, y_train, tune=tune)
    print("\nCalibrating decision thresholds on validation split")
    clf.calibrate_thresholds(X_val, y_val)
    metrics = clf.evaluate(X_test, y_test, results_dir="results")
    clf.save_models("models")
    return metrics


def parse_args() -> argparse.Namespace:
    cpu_count = os.cpu_count() or 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alaska-dir", default=DEFAULT_ALASKA_DIR)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--n", type=int, default=0, help="Number of covers; 0 means all 20,000")
    parser.add_argument("--size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count - 1))
    parser.add_argument("--tune", action="store_true", help="Run a modest hyperparameter search")
    parser.add_argument("--train-only", action="store_true", help="Reuse cached processed features")
    parser.add_argument("--force-stego", action="store_true", help="Regenerate stego PNGs")
    parser.add_argument("--force-features", action="store_true", help="Re-extract cached features")
    parser.add_argument("--smoke", action="store_true", help="Fast end-to-end check with 64 covers")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        args.n = 64
        args.force_features = True
    n_images = args.n or None

    if args.train_only:
        run_train_only(tune=args.tune, workers=args.workers)
    else:
        run_full_pipeline(
            alaska_dir=args.alaska_dir,
            output_dir=args.output_dir,
            n_images=n_images,
            img_size=args.size,
            tune=args.tune,
            workers=args.workers,
            force_stego=args.force_stego,
            force_features=args.force_features,
        )
