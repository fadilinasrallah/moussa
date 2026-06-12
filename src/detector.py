"""High-level image/video detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from classifier import Classifier
from feature_extractor import FeatureExtractor


SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv"}
TARGET_SIZE = 512


class SteganographyDetector:
    def __init__(self, model_name: str = "rf", target_size: int = TARGET_SIZE):
        self.model_name = model_name
        self.target_size = target_size
        self.extractor = FeatureExtractor()
        self.clf = Classifier()
        self._loaded = False

    def load(self, model_dir: str = "models") -> None:
        self.clf.load_models(model_dir)
        self._loaded = True

    def detect(self, path: str) -> dict:
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Call load('models') first.")

        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")

        suffix = target.suffix.lower()
        if suffix in SUPPORTED_IMAGE_FORMATS:
            return self._detect_image(target)
        if suffix in SUPPORTED_VIDEO_FORMATS:
            return self._detect_video(target)
        raise ValueError(f"Unsupported format: {suffix}")

    def _detect_image(self, image_path: Path) -> dict:
        img = self._load_rgb(image_path)
        features = self.extractor.extract(img).reshape(1, -1)
        svm_proba = float(self.clf.predict_proba(features, "svm")[0])
        rf_proba = float(self.clf.predict_proba(features, "rf")[0])
        proba = svm_proba if self.model_name == "svm" else rf_proba
        threshold = self.clf.thresholds.get(self.model_name, 0.5)
        label = int(proba >= threshold)
        result = self._format_result(label, proba, image_path)
        result["model"] = self.model_name
        result["detection_method"] = "machine_learning"
        result["feature_count"] = int(features.shape[1])
        result["svm_stego_probability"] = round(svm_proba, 4)
        result["rf_stego_probability"] = round(rf_proba, 4)
        result["svm_threshold"] = round(self.clf.thresholds.get("svm", 0.5), 4)
        result["rf_threshold"] = round(self.clf.thresholds.get("rf", 0.5), 4)
        result["model_agreement"] = (
            svm_proba >= self.clf.thresholds.get("svm", 0.5)
        ) == (
            rf_proba >= self.clf.thresholds.get("rf", 0.5)
        )
        return result

    def _detect_video(self, video_path: Path) -> dict:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV is required for video analysis: pip install opencv-python") from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        frame_interval = max(1, int(fps))
        labels = []
        probas = []
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (self.target_size, self.target_size))
                features = self.extractor.extract(frame_rgb).reshape(1, -1)
                labels.append(int(self.clf.predict(features, self.model_name)[0]))
                probas.append(float(self.clf.predict_proba(features, self.model_name)[0]))
            frame_idx += 1
        cap.release()

        if not labels:
            return {"error": "No frames analyzed"}

        label = int(np.mean(labels) >= 0.5)
        proba = float(np.mean(probas))
        result = self._format_result(label, proba, video_path)
        result["frames_analyzed"] = len(labels)
        result["stego_frame_percent"] = round(float(np.mean(labels) * 100), 2)
        result["model"] = self.model_name
        return result

    def _load_rgb(self, path: Path) -> np.ndarray:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.width < 32 or img.height < 32:
                raise ValueError("Image is too small; minimum size is 32x32")
            if img.size != (self.target_size, self.target_size):
                img = img.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
            return np.asarray(img, dtype=np.uint8)

    @staticmethod
    def _format_result(label: int, stego_probability: float, path: Path) -> dict:
        if label == 1:
            verdict = "Steganography detected"
            confidence = stego_probability
        else:
            verdict = "Clean image"
            confidence = 1.0 - stego_probability

        return {
            "file": str(path),
            "label": label,
            "stego_probability": round(stego_probability, 4),
            "confidence": round(float(confidence), 4),
            "verdict": verdict,
        }
