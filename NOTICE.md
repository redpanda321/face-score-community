# Notice: third-party terms

## This repository's code

Released under the MIT License (see `LICENSE`). The code in `app/`, `web/`, `scripts/` and `tests/`
is an independent implementation. **It contains no code copied from the SCUT-FBP5500 repository.**

## SCUT-FBP5500 (dataset, pretrained models, algorithm recipe)

The training recipe in `scripts/train.py` and the checkpoint layout accepted by `app/model.py` follow
the SCUT-FBP5500 benchmark from the Human Computer Intelligent Interaction Lab, South China University
of Technology:

- Repository: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
- Paper: Liang, Lin, Jin, Xie, Li. *SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial
  Beauty Prediction.* ICPR 2018.

Their README states:

> The SCUT-FBP5500 database can be only used for non-commercial research purpose.
> This AI algorithm is purely for academic research purpose. The dataset and codes are for academic
> research use only.

What that means here:

1. **No dataset images, labels or pretrained weights are included in this repository, and you must not
   commit them.** `.gitignore` excludes `weights/`, `data/` and `*.pth`.
2. If you obtain the dataset or weights, **you** are bound by the authors' terms. Use them for
   **non-commercial research only**. A model trained on or converted from that data inherits the same limit.
3. The MIT licence above covers this repository's code only. It does **not** grant any rights to the
   SCUT-FBP5500 data or to weights derived from it.
4. If you need a commercial deployment, do not use SCUT-derived weights. Train on data you have the rights
   to (for example photos plus ratings you collected with consent), using the same `scripts/train.py`.
5. Please cite the SCUT-FBP5500 paper if you use the dataset or its models.

## Geometric baseline

With no weights configured the app falls back to a landmark-geometry scorer (`app/geometry.py`). Its
calibration constants were fitted on AI-generated (StyleGAN2) faces that do not depict real people. That
face set is heavily skewed toward one demographic, so treat the baseline as a rough demo, not a measurement.

## Other components

MediaPipe (Apache-2.0), PyTorch / torchvision (BSD-3-Clause), FastAPI (MIT), OpenCV (Apache-2.0).
