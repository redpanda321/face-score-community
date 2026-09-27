"""Score a rated image list with the app's predictor and report Pearson / MAE / RMSE.

One fixed protocol for comparing scorers (geometric baseline vs a CNN checkpoint) on the same
held-out set. List format is SCUT-style, one "<image file> <score>" per line, scores on any scale
(Pearson is scale-free; MAE/RMSE are only meaningful when the scale is 1-5).

    python scripts/evaluate.py --data-root path/to/images --list ratings.txt                 # baseline
    python scripts/evaluate.py --data-root path/to/images --list ratings.txt --weights w.pth # CNN
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.detection import FaceDetectionError  # noqa: E402
from app.predictor import BeautyPredictor  # noqa: E402


def evaluate(predictor: BeautyPredictor, root: str, list_file: str) -> dict[str, float]:
    with open(list_file, encoding="utf-8") as handle:
        rows = [line.split() for line in handle if line.strip()]
    predicted, truth, skipped = [], [], 0
    for name, score in rows:
        try:
            with open(os.path.join(root, name), "rb") as image:
                predicted.append(predictor.predict(image.read())["score"])
            truth.append(float(score))
        except FaceDetectionError:
            skipped += 1
    pred, label = np.array(predicted), np.array(truth)
    return {
        "mode": predictor.mode,
        "n": len(pred),
        "skipped": skipped,
        "pearson": float(np.corrcoef(pred, label)[0, 1]),
        "spearman": float(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(label)))[0, 1]),
        "mae": float(np.mean(np.abs(pred - label))),
        "rmse": float(np.sqrt(np.mean((pred - label) ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--weights", help="CNN checkpoint; omit for the geometric baseline")
    args = parser.parse_args()
    result = evaluate(BeautyPredictor(args.weights), args.data_root, args.list)
    print("mode {mode}  n={n} (skipped {skipped})  Pearson {pearson:.3f}  Spearman {spearman:.3f}  MAE {mae:.3f}  RMSE {rmse:.3f}".format(**result))


if __name__ == "__main__":
    main()
