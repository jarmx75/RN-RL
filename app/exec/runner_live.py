import argparse
import yaml
import time
import os
from app.iq_client import make_client_from_env
from app.data_live import RealTimeDataFeed
from app.utils.logger import get_logger
from app.utils.storage import append_trade_log
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    logger = get_logger("runner_live")
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    client = make_client_from_env()
    assets = cfg["assets"]
    timeframes = cfg["timeframes"]
    threshold = cfg["threshold"]
    amount = cfg["amount"]
    feed = RealTimeDataFeed(client, assets, timeframes)
    feed.start()

    def predict(obs):
        probs = np.array([0.6, 0.2, 0.2])
        return np.argmax(probs), probs

    n_candles = 10
    try:
        while True:
            for asset in assets:
                for tf in timeframes:
                    dq = feed.deques[asset][tf]
                    if len(dq) < n_candles:
                        continue
                    obs = np.array([[c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in list(dq)[-n_candles:]])
                    action, probs = predict(obs)
                    if action > 0 and probs[action] >= threshold:
                        direction = "call" if action == 1 else "put"
                        ok, trade_id = client.buy(amount, asset, direction, tf // 60)
                        outcome = client.check_win_v2(trade_id)
                        append_trade_log({
                            "time": time.time(),
                            "asset": asset,
                            "tf": tf,
                            "direction": direction,
                            "prob": probs[action],
                            "amount": amount,
                            "payout": outcome,
                        })
                        logger.info(f"Trade: {asset}/{tf} {direction} → {outcome}")
            time.sleep(1)
    except KeyboardInterrupt:
        feed.stop()

if __name__ == "__main__":
    main()