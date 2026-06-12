"""
Path-based dataset pipeline for ALASKA cover/stego pairs.

The old pipeline loaded all images into memory. That is not appropriate for
20,000 512x512 covers plus their generated stego twins, so this version splits
labels first and extracts features directly from image paths in parallel.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

from feature_extractor import FeatureExtractor


def _read_label_file(labels_path: Path) -> list[dict]:
    with labels_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload["items"] if isinstance(payload, dict) else payload


def _extract_one(path: str, img_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.uint8)
    return FeatureExtractor().extract(arr)


class DatasetPipeline:
    """Split labels by source image group and extract feature matrices."""

    def __init__(self, labels_path: str, img_size: int = 512, workers: int = -1):
        self.labels_path = Path(labels_path)
        self.img_size = img_size
        self.workers = workers

    def split(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42,
    ) -> dict[str, list[dict]]:
        labels = _read_label_file(self.labels_path)
        if not labels:
            raise ValueError(f"No labels found in {self.labels_path}")

        y = np.array([row["label"] for row in labels], dtype=np.int32)
        groups = np.array([row.get("group", Path(row["file"]).stem) for row in labels])

        first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(first.split(np.zeros(len(labels)), y, groups))

        val_relative = val_size / (1.0 - test_size)
        trainval_labels = [labels[i] for i in trainval_idx]
        y_trainval = y[trainval_idx]
        groups_trainval = groups[trainval_idx]

        second = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed + 1)
        train_rel, val_rel = next(
            second.split(np.zeros(len(trainval_labels)), y_trainval, groups_trainval)
        )

        splits = {
            "train": [trainval_labels[i] for i in train_rel],
            "val": [trainval_labels[i] for i in val_rel],
            "test": [labels[i] for i in test_idx],
        }

        for name, rows in splits.items():
            yy = np.array([r["label"] for r in rows])
            unique_groups = len({r.get("group", Path(r["file"]).stem) for r in rows})
            print(
                f"{name:>5}: {len(rows)} images, {unique_groups} groups, "
                f"clean={int(np.sum(yy == 0))}, stegano={int(np.sum(yy == 1))}"
            )

        overlap = (
            {r["group"] for r in splits["train"]}
            & {r["group"] for r in splits["val"]}
            | {r["group"] for r in splits["train"]}
            & {r["group"] for r in splits["test"]}
            | {r["group"] for r in splits["val"]}
            & {r["group"] for r in splits["test"]}
        )
        if overlap:
            raise RuntimeError(f"Group leakage detected in split: {list(overlap)[:5]}")

        return splits

    def extract_split_features(
        self,
        rows: list[dict],
        name: str,
        output_dir: str = "data/processed",
        force: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        x_path = out / f"features_{name}.npy"
        y_path = out / f"y_{name}.npy"

        files_path = out / f"{name}_files.json"
        if not force and x_path.exists() and y_path.exists() and files_path.exists():
            with files_path.open(encoding="utf-8") as f:
                cached_rows = json.load(f)
            cached_signature = [(row["file"], int(row["label"])) for row in cached_rows]
            current_signature = [(row["file"], int(row["label"])) for row in rows]
            if cached_signature == current_signature:
                print(f"Loading cached {name} features from {out}")
                return np.load(x_path), np.load(y_path)
            print(f"Ignoring stale {name} feature cache; split inputs changed")

        files = [row["file"] for row in rows]
        labels = np.array([row["label"] for row in rows], dtype=np.int32)
        print(f"Extracting {name} features from {len(files)} images with workers={self.workers}")

        features = joblib.Parallel(n_jobs=self.workers, prefer="processes", batch_size=16, verbose=10)(
            joblib.delayed(_extract_one)(path, self.img_size) for path in files
        )
        X = np.asarray(features, dtype=np.float32)

        np.save(x_path, X)
        np.save(y_path, labels)
        with files_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

        return X, labels

    def build_features(
        self,
        output_dir: str = "data/processed",
        seed: int = 42,
        force: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        splits = self.split(seed=seed)
        X_train, y_train = self.extract_split_features(splits["train"], "train", output_dir, force)
        X_val, y_val = self.extract_split_features(splits["val"], "val", output_dir, force)
        X_test, y_test = self.extract_split_features(splits["test"], "test", output_dir, force)
        return X_train, X_val, X_test, y_train, y_val, y_test
