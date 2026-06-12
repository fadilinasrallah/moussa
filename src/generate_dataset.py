"""
Dataset preparation for the local ALASKA v2 JPEG 512 QF95 color covers.

The public ALASKA2 challenge used cover images plus stego variants generated with
JMiPOD, J-UNIWARD, and UERD. The folder in this project is the processed cover
set only, so this module builds a supervised dataset by pairing each real ALASKA
cover with a PNG image produced by this app's LSB embedder.
"""

from __future__ import annotations

import json
import random
import string
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

from lsb_steganography import LSBSteganography


DEFAULT_ALASKA_DIR = "ALASKA_v2_JPG_512_QF95_COLOR"
DEFAULT_TARGET_SIZE = 512


def _iter_images(dataset_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in dataset_dir.iterdir() if p.suffix.lower() in suffixes)


def _random_message(capacity_chars: int, rng: random.Random, min_chars: int) -> str:
    """Generate a varied training payload, including short app-like messages."""
    alphabet = string.ascii_letters + string.digits + " .,;:!?-_/@#$%&()[]{}"
    buckets = [
        (min_chars, 64),
        (64, 512),
        (512, 4_096),
        (4_096, 16_384),
        (16_384, max(16_385, int(capacity_chars * 0.45))),
    ]
    lo, hi = rng.choice(buckets)
    hi = min(hi, max(min_chars, capacity_chars - len(LSBSteganography.END_MARKER) - 4))
    lo = min(lo, hi)
    length = rng.randint(lo, hi)
    return "".join(rng.choices(alphabet, k=length))


def _make_pair(task: tuple[str, str, str, int, int, bool]) -> tuple[dict, dict]:
    src, clean_dst, stego_dst, target_size, seed, force = task
    src_path = Path(src)
    clean_path = Path(clean_dst)
    stego_path = Path(stego_dst)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    stego_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    lsb = LSBSteganography()
    capacity = (target_size * target_size * 3) // 8
    message = _random_message(capacity, rng, min_chars=1)

    if force or not clean_path.exists():
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            if img.size != (target_size, target_size):
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            img.save(clean_path, format="PNG")

    if force or not stego_path.exists():
        psnr = lsb.embed(str(clean_path), message, str(stego_path))
    else:
        psnr = None

    clean_label = {
        "file": str(clean_path),
        "label": 0,
        "class": "clean",
        "group": src_path.stem,
        "source": str(src_path),
        "prepared": "png",
    }
    stego_label = {
        "file": str(stego_path),
        "label": 1,
        "class": "stegano",
        "group": src_path.stem,
        "source": str(src_path),
        "prepared": "png",
        "payload_chars": len(message),
        "psnr": psnr,
    }
    return clean_label, stego_label


def prepare_alaska_lsb_dataset(
    alaska_dir: str = DEFAULT_ALASKA_DIR,
    output_dir: str = "data",
    n_images: int | None = None,
    target_size: int = DEFAULT_TARGET_SIZE,
    seed: int = 42,
    workers: int | None = None,
    force: bool = False,
) -> str:
    """
    Build labels.json using real ALASKA covers and generated LSB stego twins.

    Both classes are stored as lossless PNGs after the same RGB resize/export
    preprocessing. That prevents the classifier from learning a JPEG-vs-PNG
    shortcut instead of LSB artifacts.
    """
    dataset_dir = Path(alaska_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"ALASKA dataset directory not found: {dataset_dir}")

    images = _iter_images(dataset_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {dataset_dir}")

    rng = random.Random(seed)
    if n_images is not None and n_images > 0 and n_images < len(images):
        images = sorted(rng.sample(images, n_images))

    out = Path(output_dir)
    clean_dir = out / "raw" / "clean_alaska_png"
    stego_dir = out / "raw" / "stegano_alaska_lsb"
    labels_path = out / "labels.json"
    out.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    stego_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for stale in clean_dir.glob("*.png"):
            stale.unlink()
        for stale in stego_dir.glob("*.png"):
            stale.unlink()

    tasks = [
        (
            str(path),
            str(clean_dir / f"{path.stem}_clean.png"),
            str(stego_dir / f"{path.stem}_lsb.png"),
            target_size,
            seed + idx * 104_729,
            force,
        )
        for idx, path in enumerate(images)
    ]

    print(f"Preparing {len(images)} ALASKA cover/stego pairs at {target_size}x{target_size}")
    print(f"Clean output: {clean_dir}")
    print(f"Stego output: {stego_dir}")

    clean_labels: list[dict] = []
    stego_labels: list[dict] = []
    max_workers = workers or None
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_make_pair, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            clean_label, stego_label = future.result()
            clean_labels.append(clean_label)
            stego_labels.append(stego_label)
            if idx == 1 or idx % 250 == 0 or idx == len(futures):
                print(f"  pairs prepared: {idx}/{len(futures)}")

    clean_labels.sort(key=lambda row: row["group"])
    stego_labels.sort(key=lambda row: row["group"])
    labels = clean_labels + stego_labels
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "source": str(dataset_dir),
                    "target_size": target_size,
                    "pairs": len(images),
                    "classes": {"clean": 0, "stegano": 1},
                    "dataset_version": "prepared_png_pair_v2",
                    "note": "Clean and stego classes are both prepared PNGs from ALASKA covers; stego images use this app's LSB embedder.",
                },
                "items": labels,
            },
            f,
            indent=2,
        )

    print(f"Labels written to {labels_path}")
    return str(labels_path)


def load_labels(labels_path: str | Path) -> list[dict]:
    with Path(labels_path).open(encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload["items"]
    return payload


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Prepare ALASKA cover/stego LSB dataset")
    parser.add_argument("--alaska-dir", default=DEFAULT_ALASKA_DIR)
    parser.add_argument("--output", default="data")
    parser.add_argument("--n", type=int, default=0, help="Number of cover images; 0 means all")
    parser.add_argument("--size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prepare_alaska_lsb_dataset(
        alaska_dir=args.alaska_dir,
        output_dir=args.output,
        n_images=args.n or None,
        target_size=args.size,
        seed=args.seed,
        workers=args.workers,
        force=args.force,
    )
