"""Feature extraction for LSB steganography detection on RGB images."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from skimage.feature import local_binary_pattern
from skimage.feature.texture import graycomatrix, graycoprops


class FeatureExtractor:
    """Extract compact RGB texture and LSB statistics for binary detection."""

    def __init__(
        self,
        lbp_radius: int = 1,
        lbp_n_points: int = 8,
        glcm_distances: list[int] | None = None,
        glcm_angles: list[float] | None = None,
        n_bins_hist: int = 32,
    ):
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.glcm_distances = glcm_distances or [1, 2]
        self.glcm_angles = glcm_angles or [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        self.n_bins_hist = n_bins_hist

    def extract(self, img: np.ndarray) -> np.ndarray:
        img_rgb = self._ensure_rgb(img)
        features = [
            self._extract_lsb_header_features(img_rgb),
            self._extract_lsb_features(img_rgb),
        ]

        for channel_idx in range(3):
            features.append(self._extract_lbp_channel(img_rgb[:, :, channel_idx]))

        gray = self._rgb_to_gray(img_rgb)
        features.append(self._extract_glcm_channel(gray))
        features.append(self._extract_glcm_channel(img_rgb[:, :, 1]))

        for channel_idx in range(3):
            features.append(self._extract_histogram_channel(img_rgb[:, :, channel_idx]))

        features.append(self._extract_statistical_moments(img_rgb))
        out = np.concatenate(features).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _extract_lsb_header_features(self, img_rgb: np.ndarray) -> np.ndarray:
        """Detect the 32-bit length header used by this app's LSB embedder."""
        flat_lsb = (img_rgb.reshape(-1) & 1).astype(np.uint8)
        if flat_lsb.size < 32:
            return np.zeros(4, dtype=np.float32)

        msg_len = 0
        for bit in flat_lsb[:32]:
            msg_len = (msg_len << 1) | int(bit)

        capacity = int(flat_lsb.size - 32)
        valid = 1.0 if 0 < msg_len <= capacity else 0.0
        normalized_len = min(msg_len, capacity) / max(capacity, 1)
        header_ones = float(flat_lsb[:32].mean())
        sample = flat_lsb[32 : min(flat_lsb.size, 544)]
        payload_ones = float(sample.mean()) if sample.size else 0.0
        return np.array([valid, normalized_len, header_ones, payload_ones], dtype=np.float32)

    def _extract_lsb_features(self, img_rgb: np.ndarray) -> np.ndarray:
        features = []
        for channel_idx in range(3):
            channel = img_rgb[:, :, channel_idx].astype(np.int32)
            lsb_plane = channel & 1
            n_pixels = lsb_plane.size
            n_ones = int(lsb_plane.sum())
            ratio = n_ones / max(n_pixels, 1)
            features.append(ratio)

            vals = channel.ravel()
            even_vals = vals[vals % 2 == 0]
            odd_vals = vals[vals % 2 == 1]
            bins = np.arange(0, 129)
            hist_even, _ = np.histogram(even_vals // 2, bins=bins)
            hist_odd, _ = np.histogram(odd_vals // 2, bins=bins)
            denom = hist_even + hist_odd + 1e-10
            chi2 = np.sum((hist_even - hist_odd) ** 2 / denom) / (128 * max(n_pixels, 1))
            features.append(float(chi2))

            features.append(float(np.var(lsb_plane.astype(np.float32))))
            flat = lsb_plane.ravel().astype(np.float32)
            if flat.size > 1 and np.std(flat[:-1]) > 0 and np.std(flat[1:]) > 0:
                autocorr = float(np.corrcoef(flat[:-1], flat[1:])[0, 1])
            else:
                autocorr = 0.0
            features.append(autocorr)

        return np.array(features, dtype=np.float32)

    def _extract_lbp_channel(self, channel: np.ndarray) -> np.ndarray:
        channel_u8 = self._to_uint8(channel)
        lbp = local_binary_pattern(
            channel_u8, self.lbp_n_points, self.lbp_radius, method="uniform"
        )
        n_bins = self.lbp_n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist.astype(np.float32)

    def _extract_glcm_channel(self, channel: np.ndarray) -> np.ndarray:
        channel_u8 = self._to_uint8(channel)
        img_q = (channel_u8 // 4).astype(np.uint8)
        glcm = graycomatrix(
            img_q,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=64,
            symmetric=True,
            normed=True,
        )
        features = []
        for prop in ["contrast", "correlation", "energy", "homogeneity"]:
            features.extend(graycoprops(glcm, prop).ravel().tolist())
        return np.array(features, dtype=np.float32)

    def _extract_histogram_channel(self, channel: np.ndarray) -> np.ndarray:
        channel_u8 = self._to_uint8(channel)
        hist, _ = np.histogram(
            channel_u8.ravel(), bins=self.n_bins_hist, range=(0, 256), density=True
        )
        return hist.astype(np.float32)

    def _extract_statistical_moments(self, img_rgb: np.ndarray) -> np.ndarray:
        features = []
        for channel_idx in range(3):
            ch = img_rgb[:, :, channel_idx].ravel().astype(np.float32)
            features.extend(
                [
                    float(ch.mean() / 255.0),
                    float(ch.std() / 255.0),
                    float(sp_stats.skew(ch)),
                    float(sp_stats.kurtosis(ch)),
                ]
            )
        return np.array(features, dtype=np.float32)

    def extract_batch(self, X: np.ndarray) -> np.ndarray:
        features = []
        for idx, img in enumerate(X, start=1):
            features.append(self.extract(img))
            if idx == 1 or idx % 100 == 0 or idx == len(X):
                print(f"  features extracted: {idx}/{len(X)}")
        return np.array(features, dtype=np.float32)

    @staticmethod
    def _ensure_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img_u8 = FeatureExtractor._to_uint8(img)
            return np.stack([img_u8, img_u8, img_u8], axis=-1)
        if img.ndim == 3 and img.shape[2] == 3:
            return FeatureExtractor._to_uint8(img)
        raise ValueError(f"Unsupported image shape: {img.shape}")

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        if img.size and float(np.nanmax(img)) <= 1.0:
            return np.clip(img * 255, 0, 255).astype(np.uint8)
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
        return (
            0.299 * img_rgb[:, :, 0]
            + 0.587 * img_rgb[:, :, 1]
            + 0.114 * img_rgb[:, :, 2]
        ).clip(0, 255).astype(np.uint8)
