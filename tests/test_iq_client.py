from pathlib import Path

import pytest

from app.iq_client import DryRunIQClient, make_client_from_env


def test_make_client_from_env_without_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {"assets": ["EURUSD", "GBPUSD"], "timeframes": [60, 300]}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("assets: [EURUSD, GBPUSD]\ntimeframes: [60, 300]\n", encoding="utf-8")

    monkeypatch.delenv("IQ_EMAIL", raising=False)
    monkeypatch.delenv("IQ_PASSWORD", raising=False)
    monkeypatch.delenv("IQ_ACCOUNT", raising=False)

    client = make_client_from_env(str(cfg_path))
    assert isinstance(client, DryRunIQClient)


def test_dryrun_stream_and_trade(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure randomness is deterministic for assertions
    monkeypatch.setenv("PYTHONHASHSEED", "0")

    client = DryRunIQClient(["EURUSD"], [60])
    client.start_candles_stream("EURUSD", 60, 5)
    candles = client.get_realtime_candles("EURUSD", 60)

    assert len(candles) == 5
    for candle in candles:
        assert {"open", "close", "high", "low", "volume", "epoch"} <= candle.keys()

    ok, trade_id = client.buy(10, "EURUSD", "call", 1)
    assert ok is True
    assert trade_id.startswith("DRY_")

    result = client.check_win_v2(trade_id, poll_sec=0)
    assert isinstance(result, float)


def test_dryrun_open_time_structure() -> None:
    client = DryRunIQClient(["EURUSD", "GBPUSD"], [60, 300])
    open_time = client.get_all_open_time()
    assert set(open_time.keys()) == {"EURUSD", "GBPUSD"}
    for asset_info in open_time.values():
        assert asset_info["open"] is True
        assert set(asset_info["timeframes"].keys()) == {60, 300}

