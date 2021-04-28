from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from CryptoEnv.CryptoEnv import CryptoEnv


# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: CryptoEnv()])
#env = CryptoEnv()
#env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
env.close()
