def test_imports():
    import app.iq_client
    import app.data_live
    import app.rl.env
    from app.iq_client import DryRunIQClient
    from app.data_live import RealTimeDataFeed
    c = DryRunIQClient(["EURUSD"], [60])
    feed = RealTimeDataFeed(c, ["EURUSD"], [60])
    from app.rl.env import TradingEnv
    env = TradingEnv()
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape