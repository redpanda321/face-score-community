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
It was verified with a round-trip test, **not** against the real SCUT files, which are not downloadable
without their Baidu Pan password.

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

The script reports Pearson correlation, MAE and RMSE. It was smoke-tested on synthetic data only; it has
not been run on the real dataset, so no accuracy figure is claimed here. See the SCUT repository for
the authors' reported benchmarks.

### Tests

```bash
pip install -r requirements-dev.txt && pytest -q
```

### Limits

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

从原作者处获取数据集并遵守其条款后，运行 `scripts/train.py`（命令见上文英文部分）。脚本仅用合成数据做过冒烟测试，
未在真实数据集上运行，因此本仓库不给出任何准确率数字，请参考 SCUT 官方仓库的基准结果。

### 许可

代码采用 MIT 许可证。SCUT-FBP5500 的数据与权重不在此许可范围内，商业用途请勿使用，详见 [NOTICE.md](NOTICE.md)。
