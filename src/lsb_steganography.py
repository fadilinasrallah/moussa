"""LSB text embedding and extraction for RGB images."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


class LSBSteganography:
    """Embed UTF-8 text in the least significant bits of RGB pixels."""

    HEADER_BITS = 32
    END_MARKER = "<<<END>>>"

    def embed(self, image_path: str, message: str, output_path: str) -> float:
        img = Image.open(image_path).convert("RGB")
        img_array = np.asarray(img, dtype=np.uint8).copy()
        original = img_array.copy()

        payload = (message + self.END_MARKER).encode("utf-8")
        payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        msg_len = int(payload_bits.size)
        header_bits = np.array([int(bit) for bit in f"{msg_len:032b}"], dtype=np.uint8)
        full_bits = np.concatenate([header_bits, payload_bits])

        flat = img_array.reshape(-1)
        if full_bits.size > flat.size:
            raise ValueError(
                f"Message is too long ({full_bits.size} bits) for this image "
                f"({flat.size} bits available)."
            )

        flat[: full_bits.size] = (flat[: full_bits.size] & 0xFE) | full_bits
        psnr = self._compute_psnr(original, img_array)
        Image.fromarray(img_array).save(output_path)
        return psnr

    def extract(self, image_path: str) -> str:
        img_array = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        flat_lsb = img_array.reshape(-1) & 1
        if flat_lsb.size < self.HEADER_BITS:
            return ""

        msg_len = 0
        for bit in flat_lsb[: self.HEADER_BITS]:
            msg_len = (msg_len << 1) | int(bit)

        if msg_len <= 0 or msg_len > flat_lsb.size - self.HEADER_BITS:
            return ""

        payload_bits = flat_lsb[self.HEADER_BITS : self.HEADER_BITS + msg_len].astype(np.uint8)
        usable_len = (payload_bits.size // 8) * 8
        if usable_len == 0:
            return ""

        payload = np.packbits(payload_bits[:usable_len]).tobytes()
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            text = payload.decode("utf-8", errors="ignore")

        if self.END_MARKER in text:
            return text[: text.index(self.END_MARKER)]
        return ""

    def get_capacity(self, image_path: str) -> dict:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        bits = width * height * 3 - self.HEADER_BITS
        return {
            "width": width,
            "height": height,
            "capacity_bits": bits,
            "capacity_bytes": bits // 8,
            "capacity_chars_estimate": bits // 8,
        }

    @staticmethod
    def _compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
        mse = np.mean((original.astype(np.float32) - modified.astype(np.float32)) ** 2)
        if mse == 0:
            return float("inf")
        return 10 * math.log10((255**2) / mse)
