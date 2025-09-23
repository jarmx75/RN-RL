import os
import pandas as pd
from typing import List, Any

class HistoricalDataManager:
    def __init__(self, client, data_dir: str = "data"):
        self.client = client
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_data(self, asset: str, tf: int, days: int = 7):
        import numpy as np
        import time
        candles = []
        price = 1.0
        for _ in range(days * 1440 // (tf // 60)):
            change = np.random.normal(0, 0.001)
            o = price
            c = price + change
            h = max(o, c) + abs(np.random.normal(0, 0.0005))
            l = min(o, c) - abs(np.random.normal(0, 0.0005))
            v = np.random.randint(1, 10)
            candles.append({"open": o, "close": c, "high": h, "low": l, "volume": v, "epoch": time.time()})
            price = c
        df = pd.DataFrame(candles)
        fname = os.path.join(self.data_dir, f"{asset}_{tf}.csv")
        df.to_csv(fname, index=False)
        return fname

    def load_data(self, asset: str, tf: int) -> pd.DataFrame:
        fname = os.path.join(self.data_dir, f"{asset}_{tf}.csv")
        if os.path.isfile(fname):
            return pd.read_csv(fname)
        else:
            return pd.DataFrame()