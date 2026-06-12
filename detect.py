#!/usr/bin/env python3
"""CLI for steganography detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from detector import SteganographyDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect LSB steganography in an image or video")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--image", help="Image file to analyze")
    target.add_argument("--video", help="Video file to analyze")
    parser.add_argument("--model", choices=["svm", "rf"], default="rf")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    missing = [name for name in ["svm_model.pkl", "rf_model.pkl"] if not (models_dir / name).exists()]
    if missing:
        print(f"Missing model files in {models_dir}: {', '.join(missing)}", file=sys.stderr)
        print("Train first with: python run_pipeline.py --n 4000", file=sys.stderr)
        return 1

    detector = SteganographyDetector(model_name=args.model)
    detector.load(str(models_dir))

    try:
        result = detector.detect(args.image or args.video)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"File:        {result['file']}")
        print(f"Verdict:     {result['verdict']}")
        print(f"Confidence:  {result['confidence'] * 100:.1f}%")
        print(f"Stego prob.: {result['stego_probability'] * 100:.1f}%")
        if "frames_analyzed" in result:
            print(f"Frames:      {result['frames_analyzed']}")
            print(f"Stego frames:{result['stego_frame_percent']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
