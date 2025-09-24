"""Validación de shapes del extractor de features."""

from __future__ import annotations

import numpy as np

from app.rl.features import build_feature_tensor


def test_feature_tensor_shape():
    data = {
        ("EURUSD", 60): np.random.rand(20, 5).astype(np.float32),
        ("GBPUSD", 300): np.random.rand(20, 5).astype(np.float32),
    }
    tensor = build_feature_tensor(data)
    assert tensor.shape == (2, 20, 6)
    assert np.allclose(tensor.mean(axis=(1, 2)), 0, atol=1.0)
