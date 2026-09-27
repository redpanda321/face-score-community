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

## Geometric baseline and the London Set (CC BY 4.0)

With no weights configured the app falls back to a landmark-geometry scorer (`app/geometry.py`).

The mapping from geometry to a 1-5 score is a six-coefficient ridge regression fitted on the
**Face Research Lab London Set**:

> DeBruine, L., & Jones, B. (2021). *Face Research Lab London Set*. figshare.
> https://doi.org/10.6084/m9.figshare.5047666.v5 — licensed CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/). Modified: only summary statistics of the ratings were used.

No London Set images or ratings are redistributed here; the repository contains only the fitted coefficients.
Its participants consented to use "in lab-based and web-based studies … and to illustrate research", which is
narrower than what the CC BY licence permits. If you deploy this commercially, get your own legal advice.

Measured accuracy (nested leave-one-out on those same 102 faces): Pearson r ≈ 0.46 with mean human rating,
and only about 0.15 within male faces. That is a modest statistical fit on a small, mostly-White, studio-lit
sample, not a validated measurement. The unit-scale constants used to normalise the geometric measurements
(`IDEAL`, `TOLERANCES` in `app/geometry.py`) were derived from AI-generated (StyleGAN2) faces that depict no
real people; that set is skewed toward one demographic.

## Other components

MediaPipe (Apache-2.0), PyTorch / torchvision (BSD-3-Clause), FastAPI (MIT), OpenCV (Apache-2.0).
