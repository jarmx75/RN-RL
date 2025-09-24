"""Transformaciones de features para velas multi-activo."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

CandleArray = np.ndarray
Key = Tuple[str, int]


def compute_returns(prices: np.ndarray) -> np.ndarray:
    returns = np.diff(np.log(prices + 1e-8), prepend=prices[0])
    return returns.astype(np.float32)


def normalize(array: np.ndarray) -> np.ndarray:
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True) + 1e-6
    return ((array - mean) / std).astype(np.float32)


def build_feature_tensor(data: Dict[Key, CandleArray]) -> np.ndarray:
    """Concatena features multi-activo/multi-timeframe.

    Parameters
    ----------
    data:
        Diccionario (asset, timeframe) -> array shape (window, features).

    Returns
    -------
    np.ndarray
        Tensor con shape (num_pairs, window, feature_dim).
    """

    tensors = []
    for key in sorted(data.keys()):
        candles = data[key]
        close = candles[:, 1]
        returns = compute_returns(close)
        highs = candles[:, 2]
        lows = candles[:, 3]
        volumes = candles[:, 4]
        stacked = np.stack(
            [
                candles[:, 0],  # open
                close,
                highs,
                lows,
                volumes,
                returns,
            ],
            axis=-1,
        )
        tensors.append(normalize(stacked))
    if not tensors:
        raise ValueError("No hay datos suficientes para construir features")
    return np.stack(tensors, axis=0)
