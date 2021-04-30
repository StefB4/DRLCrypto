from gym.envs.registration import register
from cryptoenv.envs.CryptoEnv import CryptoEnv

register(
    id='cryptoenv-v0',
    entry_point='cryptoenv.envs:CryptoEnv',
)