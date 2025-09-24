from app.rl.env import TradingEnv


def test_env_shapes():
    assets = ["EURUSD", "GBPUSD"]
    timeframes = [60, 300]
    window_size = 32
    env = TradingEnv(assets=assets, timeframes=timeframes, window_size=window_size, episode_length=32)

    expected_shape = (len(assets), len(timeframes), window_size, 6)
    assert env.observation_space.shape == expected_shape
    assert env.action_space.n == 3

    obs, info = env.reset()
    assert obs.shape == expected_shape
    assert set(info.keys()) >= {"asset", "timeframe"}

    next_obs, reward, done, truncated, info = env.step(0)
    assert next_obs.shape == expected_shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)