"""ResNet-18 facial beauty regressor, loadable from SCUT-FBP5500-style checkpoints.

The architecture is a plain ResNet-18 with a single regression output, the same
recipe the SCUT-FBP5500 paper describes (224x224 input, L2 loss, scores in [1, 5]).
This file is an independent implementation: it contains no code from the
SCUT-FBP5500 repository, only a key-name converter so checkpoints trained with
their layout (``group1.*`` / ``group2.fullyconnected.*``) load into it.

Weights are NOT bundled. See NOTICE.md for the licence terms that apply to them.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

RESIZE = 256
CROP = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_SCORE, MAX_SCORE = 1.0, 5.0


def build_model() -> nn.Module:
    """A ResNet-18 with one regression output and no pretrained weights."""
    return resnet18(weights=None, num_classes=1)


def convert_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Rename SCUT-style checkpoint keys to the torchvision ResNet-18 layout.

    Handles a DataParallel ``module.`` prefix, the ``group1.`` wrappers around the
    stem and residual-block layers, and ``group2.fullyconnected`` for the head.
    Keys already in torchvision layout pass through unchanged.
    """
    converted: dict[str, Any] = {}
    for key, value in state.items():
        key = re.sub(r"^module\.", "", key)
        key = key.replace("group2.fullyconnected", "fc")
        key = key.replace("group1.", "")
        converted[key] = value
    return converted


def load_weights(path: str, device: str = "cpu") -> nn.Module:
    """Load a checkpoint into a fresh model, failing loudly on any mismatch."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = build_model()
    result = model.load_state_dict(convert_state_dict(state), strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "Checkpoint does not match ResNet-18 "
            f"(missing={sorted(result.missing_keys)[:5]}, "
            f"unexpected={sorted(result.unexpected_keys)[:5]})"
        )
    return model.to(device).eval()


def preprocess(face_rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB face crop -> 1x3x224x224 normalised tensor.

    Resize to 256 then centre-crop 224, matching the training recipe.
    """
    from PIL import Image

    image = Image.fromarray(face_rgb).resize((RESIZE, RESIZE), Image.BILINEAR)
    offset = (RESIZE - CROP) // 2
    cropped = np.asarray(image, dtype=np.float32)[offset : offset + CROP, offset : offset + CROP] / 255.0
    normalised = (cropped - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(normalised.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def predict_score(model: nn.Module, face_rgb: np.ndarray) -> float:
    """Predict a beauty score in [1, 5] for one face crop."""
    raw = float(model(preprocess(face_rgb)).squeeze())
    return round(float(np.clip(raw, MIN_SCORE, MAX_SCORE)), 2)
