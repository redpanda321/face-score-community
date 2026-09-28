# Face Score Community

[English](#english) · [中文](#中文)

## English

A free, self-hosted facial-beauty scoring API and web page on a **1–5 scale**, following the
[SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) benchmark recipe
(ResNet-18 regression, 224×224 input, L2 loss). No accounts, no payments, no telemetry; uploaded photos
are scored in memory and never stored.

> **Read this first.**
> - **No model weights are included.** The SCUT-FBP5500 weights are distributed by their authors and are
>   limited to **non-commercial research use** (see [NOTICE.md](NOTICE.md)). Bring your own checkpoint, or
>   train one from the dataset with `scripts/train.py`.
> - Without weights the app runs a **geometric baseline**. It is a rough demo, not a measurement, and the
>   UI labels it as such.
> - Scores model *average rater preference on a research dataset*. They are not a judgement of anyone's
>   worth or attractiveness. Only upload photos of yourself or people who consented; do not use it on minors.

### Two modes

| Mode | When | What it does |
|---|---|---|
| `cnn` | `MODEL_PATH` points to a ResNet-18 checkpoint | Crops the face, predicts a 1–5 score |
| `geometry` | no `MODEL_PATH` | Landmark-proportion baseline mapped to 1–5 |

`app/model.py` loads plain checkpoints and SCUT-style ones (`state_dict` wrapper, `module.` prefix,
`group1.*` / `group2.fullyconnected.*` key names). Loading is strict, so a mismatched file fails loudly.
Checkpoints produced by `scripts/train.py` are loaded and scored end to end in the test above. The
SCUT-style key-name remapping is verified only by a round-trip test, **not** against the authors' own
weight files, which are not downloadable without their Baidu Pan password.

### Run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

MODEL_PATH=weights/resnet18.pth uvicorn app.main:app --port 8000   # cnn mode
uvicorn app.main:app --port 8000                                   # geometry baseline
```

Open http://localhost:8000, or:

```bash
curl -F file=@face.jpg http://localhost:8000/api/score
# {"score": 3.4, "mode": "cnn", "notice": "..."}
```

Docker: `docker build -t face-score-community .` then
`docker run -p 8000:8000 -v $PWD/weights:/weights -e MODEL_PATH=/weights/resnet18.pth face-score-community`
(the Dockerfile has not been build-tested yet).

### API

| Endpoint | Description |
|---|---|
| `POST /api/score` | multipart `file` → `{score, mode, ...}`; errors are `400 {code, detail}` (`no_face`, `multiple_faces`, `image_too_small`, `image_unreadable`, `file_too_large`) |
| `GET /api/info` | current mode, scale and notice |
| `GET /health` | liveness |

### Train your own

Get the dataset from its authors and accept its terms, then:

```bash
python scripts/train.py --data-root data/faces --train-list data/1/train_1.txt \
    --test-list data/1/test_1.txt --out weights/resnet18.pth
python scripts/train.py --data-root data/faces --test-list data/1/test_1.txt --eval-only --weights weights/resnet18.pth
```

The script reports Pearson correlation, MAE and RMSE, and can checkpoint (`--save-every`) and resume
(`--resume`) long runs.

**Measured result** (this repo's `scripts/train.py`, ImageNet-pretrained ResNet-18, official 60%/40% split,
20,000 iterations, 2,200 held-out test faces, no train/test file overlap): Pearson **0.898**, MAE 0.233,
RMSE 0.303 with the paper's protocol (whole image in). Through this app's real pipeline (detect the face,
crop, score) it is Pearson **0.869**, MAE 0.280, RMSE 0.362. For reference, the SCUT authors report
ResNet-18 at PC 0.851 / MAE 0.282 / RMSE 0.370 on the same split. Our run used ImageNet-pretrained weights
and horizontal-flip augmentation, which may explain the gap; no ablation was run. The run was interrupted
once and resumed from iteration 2000 (optimizer momentum was not restored). These numbers are for the SCUT
test set only: performance on smiling, casually lit or non-frontal selfies is untested.

Weights trained on SCUT-FBP5500 inherit its **non-commercial research** licence (see NOTICE.md), so they are
not distributed in this repository.

### Compare scorers on the same data

`scripts/evaluate.py` scores a rated image list (`<image> <score>` per line) and prints Pearson, Spearman,
MAE and RMSE, so the geometric baseline and any CNN checkpoint are judged by one protocol:

```bash
python scripts/evaluate.py --data-root images/ --list ratings.txt                    # geometric baseline
python scripts/evaluate.py --data-root images/ --list ratings.txt --weights w.pth    # CNN
```

Always evaluate on faces the scorer was not fitted on. Pearson is scale-free; MAE/RMSE are only meaningful
when your ratings are on a 1-5 scale.

### Tests

```bash
pip install -r requirements-dev.txt && pytest -q
```

### Limits

- The geometric baseline was fitted on 102 faces (London Set, CC BY 4.0, see [NOTICE.md](NOTICE.md)) and
  reaches Pearson r ≈ 0.46 with human ratings in nested cross-validation (≈ 0.15 within male faces). Treat
  it as a rough demo. A CNN trained on a large rated dataset is expected to do far better.
- Landmark geometry cannot see smile, skin, age or hair, which strongly affect human ratings; the
  baseline will not track them.
- Photos should be frontal, well lit, one face. Large head turns hurt the baseline.
- Rating datasets carry the demographics and tastes of their raters and subjects.

## 中文

免费、可自托管的人脸颜值评分 API 与网页，评分范围 **1–5 分**，训练方案参照
[SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)（ResNet-18 回归、224×224 输入、L2 损失）。
无需账号、无支付、无遥测，上传的照片只在内存中处理，不会保存。

> **请先阅读**
> - **本仓库不包含任何模型权重。** SCUT-FBP5500 的数据与权重由原作者发布，**仅限非商业研究使用**
> （见 [NOTICE.md](NOTICE.md)）。请自行准备权重，或用 `scripts/train.py` 基于数据集自行训练。
> - 未配置权重时，程序使用**几何比例基线**，仅作演示，界面会明确标注。
> - 分数反映的是研究数据集中评分者的平均偏好，**不代表任何人的价值或吸引力**。请仅上传本人或已获同意者的照片，不要用于未成年人。

### 运行

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
MODEL_PATH=weights/resnet18.pth uvicorn app.main:app --port 8000   # CNN 模式
uvicorn app.main:app --port 8000                                   # 几何基线模式
```

浏览器打开 http://localhost:8000，或 `curl -F file=@face.jpg http://localhost:8000/api/score`。

### 自行训练

从原作者处获取数据集并遵守其条款后，运行 `scripts/train.py`（命令见上文英文部分），支持断点保存与续训。
实测结果（官方 60%/40% 划分、2,200 张测试集）：论文同款整图输入 Pearson 0.898；经本程序完整流程（检测人脸→裁剪→打分）Pearson 0.869。
仅在 SCUT 测试集上验证过，对笑脸、随手拍、非正脸的自拍效果未测试。用 SCUT 训练出的权重同样仅限非商业研究使用，因此不随仓库分发。

### 许可

代码采用 MIT 许可证。SCUT-FBP5500 的数据与权重不在此许可范围内，商业用途请勿使用，详见 [NOTICE.md](NOTICE.md)。
