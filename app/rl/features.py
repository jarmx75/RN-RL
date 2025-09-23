import numpy as np

def normalize_ohlcv(data: np.ndarray) -> np.ndarray:
    price = data[..., :4]
    price_norm = (price - price.mean(axis=0)) / (price.std(axis=0) + 1e-8)
    v = data[..., 4:]
    v_norm = (v - v.mean()) / (v.std() + 1e-8)
    return np.concatenate([price_norm, v_norm], axis=-1)