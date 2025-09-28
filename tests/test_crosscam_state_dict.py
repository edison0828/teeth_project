"""Regression tests for CrossCamInference state dict handling."""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.COLORMAP_JET = 0
    cv2_stub.applyColorMap = lambda image, colormap: image
    cv2_stub.addWeighted = lambda src1, alpha, src2, beta, gamma: src1
    cv2_stub.imwrite = lambda path, image: True
    sys.modules["cv2"] = cv2_stub

from backend.inference import CrossCamInference
from backend.inference_models import build_cross_attn_fdi


def test_crosscam_loads_prefixed_state_dict(tmp_path) -> None:
    model = build_cross_attn_fdi(num_fdi=2)
    state_dict = model.state_dict()
    prefixed_state = {f"module.{key}": value for key, value in state_dict.items()}

    checkpoint_path = tmp_path / "classifier.pt"
    torch.save({"state_dict": prefixed_state}, checkpoint_path)

    detector_path = tmp_path / "detector.pt"
    detector_path.write_bytes(b"fake")

    inference = CrossCamInference(
        detector_path=detector_path,
        classifier_path=checkpoint_path,
    )
    inference._detector = object()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inference._ensure_models()

    relevant = [
        warning
        for warning in caught
        if not (
            warning.category is FutureWarning
            and "weights_only" in str(warning.message)
        )
    ]

    assert not relevant
    assert inference._classifier is not None
