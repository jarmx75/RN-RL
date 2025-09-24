import datetime as dt

from app.risk import RiskConfig, RiskEngine


def test_consecutive_losses_block_trading():
    config = RiskConfig(max_consecutive_losses=2, daily_stop_loss=0, daily_take_profit=0)
    engine = RiskEngine(config)
    now = dt.datetime(2024, 1, 1, 10, 0, 0)

    assert engine.can_trade(asset="EURUSD", timeframe=60, probability=0.8, now=now)
    engine.register_open("EURUSD", 60)
    engine.register_close(asset="EURUSD", timeframe=60, result=-1.0, now=now)
    engine.register_open("EURUSD", 60)
    engine.register_close(asset="EURUSD", timeframe=60, result=-1.0, now=now)

    assert engine.consecutive_losses == 2
    assert not engine.can_trade(asset="EURUSD", timeframe=60, probability=0.9, now=now)


def test_dynamic_threshold_adjustment():
    config = RiskConfig(dynamic_threshold=True, base_threshold=0.7, threshold_step=0.05)
    engine = RiskEngine(config)
    now = dt.datetime(2024, 1, 1, 11, 0, 0)

    engine.register_open("EURUSD", 60)
    engine.register_close(asset="EURUSD", timeframe=60, result=-1.0, now=now)
    assert engine.threshold == 0.75

    engine.register_open("EURUSD", 60)
    engine.register_close(asset="EURUSD", timeframe=60, result=0.8, now=now)
    assert engine.threshold == 0.7


def test_daily_reset():
    config = RiskConfig(max_consecutive_losses=1, daily_stop_loss=0, daily_take_profit=0)
    engine = RiskEngine(config)
    day1 = dt.datetime(2024, 1, 1, 9, 0, 0)
    day2 = dt.datetime(2024, 1, 2, 9, 0, 0)

    engine.register_open("EURUSD", 60)
    engine.register_close(asset="EURUSD", timeframe=60, result=-1.0, now=day1)
    assert not engine.can_trade(asset="EURUSD", timeframe=60, probability=0.8, now=day1)

    assert engine.can_trade(asset="EURUSD", timeframe=60, probability=0.8, now=day2)
