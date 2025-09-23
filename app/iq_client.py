import os
import random
import time
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

class IQClient:
    def __init__(self):
        from iqoptionapi.stable_api import IQ_Option
        self.IQ_Option = IQ_Option
        self.iq = None
        self.connected = False

    def connect(self, email: str, password: str, account: str = "PRACTICE") -> bool:
        self.iq = self.IQ_Option(email, password)
        self.iq.connect()
        if self.iq.check_connect():
            self.iq.change_balance(account)
            self.connected = True
        else:
            self.connected = False
        return self.connected

    def change_balance(self, balance_type: str):
        if self.connected:
            self.iq.change_balance(balance_type)

    def get_all_open_time(self) -> Dict[str, Any]:
        try:
            return self.iq.get_all_open_time()
        except Exception:
            return {}

    def start_candles_stream(self, asset: str, tf: int, n: int):
        self.iq.start_candles_stream(asset, tf, n)

    def get_realtime_candles(self, asset: str, tf: int) -> List[Dict[str, Any]]:
        return self.iq.get_realtime_candles(asset, tf)

    def stop_candles_stream(self, asset: str, tf: int):
        self.iq.stop_candles_stream(asset, tf)

    def buy(self, amount: float, asset: str, direction: str, minutes: int) -> Tuple[bool, Optional[str]]:
        _, id = self.iq.buy(amount, asset, direction, minutes)
        return True, id

    def check_win_v2(self, trade_id: str, poll_sec: int = 3) -> float:
        time.sleep(poll_sec)
        return self.iq.check_win_v2(trade_id)

    def check_connect(self) -> bool:
        return self.iq.check_connect()

class DryRunIQClient:
    """Simula IQ Option API para tests y desarrollo"""
    def __init__(self, assets: List[str], timeframes: List[int]):
        self.assets = assets
        self.timeframes = timeframes
        self.trades = {}
        self.balance = 1000.0

    def connect(self, email: str = "", password: str = "", account: str = "PRACTICE") -> bool:
        return True

    def change_balance(self, balance_type: str):
        return

    def get_all_open_time(self) -> Dict[str, Any]:
        return {asset: {"open": True} for asset in self.assets}

    def start_candles_stream(self, asset: str, tf: int, n: int):
        pass

    def get_realtime_candles(self, asset: str, tf: int) -> List[Dict[str, Any]]:
        import numpy as np
        candles = []
        price = 1.0 + 0.01 * random.random()
        for _ in range(10):
            change = np.random.normal(0, 0.001)
            o = price
            c = price + change
            h = max(o, c) + abs(np.random.normal(0, 0.0005))
            l = min(o, c) - abs(np.random.normal(0, 0.0005))
            v = random.randint(1, 10)
            candles.append({"open": o, "close": c, "high": h, "low": l, "volume": v, "epoch": time.time()})
            price = c
        return candles

    def stop_candles_stream(self, asset: str, tf: int):
        pass

    def buy(self, amount: float, asset: str, direction: str, minutes: int) -> Tuple[bool, str]:
        trade_id = f"DRY_{random.randint(1000,9999)}"
        self.trades[trade_id] = {"asset": asset, "direction": direction, "amount": amount}
        return True, trade_id

    def check_win_v2(self, trade_id: str, poll_sec: int = 1) -> float:
        trade = self.trades.get(trade_id, None)
        if not trade:
            return 0.0
        win = random.choice([True, False])
        return 0.8 * trade["amount"] if win else -trade["amount"]

    def check_connect(self) -> bool:
        return True

def make_client_from_env(config_path: str = "config/config.yaml"):
    load_dotenv()
    email = os.getenv("IQ_EMAIL", "")
    password = os.getenv("IQ_PASSWORD", "")
    account = os.getenv("IQ_ACCOUNT", "PRACTICE")
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    assets = cfg.get("assets", ["EURUSD", "GBPUSD"])
    timeframes = cfg.get("timeframes", [60])
    if not email or not password:
        return DryRunIQClient(assets, timeframes)
    client = IQClient()
    if client.connect(email, password, account):
        return client
    else:
        return DryRunIQClient(assets, timeframes)