import re

import numpy as np
import pytest
import torch

from app.model import build_model, convert_state_dict, load_weights, predict_score


def to_scut_layout(state: dict) -> dict:
    """Inverse of convert_state_dict: torchvision keys -> SCUT-style keys."""
    out = {}
    for key, value in state.items():
        key = re.sub(r"^(conv1|bn1)\.", r"group1.\1.", key)
        key = re.sub(r"^(layer\d\.\d)\.(conv1|bn1|conv2|bn2)\.", r"\1.group1.\2.", key)
        key = re.sub(r"^fc\.", "group2.fullyconnected.", key)
        out[key] = value
    return out


@pytest.fixture
def reference_model():
    torch.manual_seed(0)
    return build_model().eval()


def test_converter_maps_scut_keys_back_to_torchvision(reference_model):
    original = reference_model.state_dict()
    scut = to_scut_layout(original)
    assert any(k.startswith("group1.") for k in scut) and "group2.fullyconnected.weight" in scut
    assert set(convert_state_dict(scut)) == set(original)


@pytest.mark.parametrize("wrap", ["raw", "state_dict", "module_prefix"])
def test_load_weights_accepts_scut_checkpoints(tmp_path, reference_model, wrap):
    state = to_scut_layout(reference_model.state_dict())
    if wrap == "module_prefix":
        state = {f"module.{k}": v for k, v in state.items()}
    payload = {"state_dict": state} if wrap != "raw" else state
    path = tmp_path / "w.pth"
    torch.save(payload, path)

    loaded = load_weights(str(path))
    sample = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(loaded(sample), reference_model(sample), atol=1e-6)


def test_load_weights_rejects_a_mismatched_checkpoint(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"state_dict": {"not.a.resnet": torch.zeros(1)}}, path)
    with pytest.raises(ValueError):
        load_weights(str(path))


def test_predict_score_is_clipped_to_the_scale(reference_model):
    face = np.random.default_rng(0).integers(0, 255, (300, 260, 3), dtype=np.uint8)
    assert 1.0 <= predict_score(reference_model, face) <= 5.0
    reference_model.fc.bias.data.fill_(100.0)
    assert predict_score(reference_model, face) == 5.0
    reference_model.fc.bias.data.fill_(-100.0)
    assert predict_score(reference_model, face) == 1.0
