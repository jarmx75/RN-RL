def test_env_shapes():
    from app.rl.env import TradingEnv
    env = TradingEnv()
    assert env.observation_space.shape == (10,5)
    assert env.action_space.n == 3
    obs, _ = env.reset()
    action = 0
    next_obs, reward, done, _, _ = env.step(action)
    assert next_obs.shape == (10,5)
    assert isinstance(reward, float)
    assert isinstance(done, bool)