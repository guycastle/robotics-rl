import gym
import envs
import numpy as np
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import warnings
warnings.filterwarnings('ignore')

# envKwargs = {
#     'resizeCamImagePct': 50,
#     'ledHSVLower': np.array([0, 0, 252]),
#     'ledHSVHigher': np.array([31, 9, 255]),
#     'rPiIP': '192.168.0.183',
#     'rPiPort': 50000
# }

# env = make_vec_env('RPiLEDEnv-v0', n_envs=1, env_kwargs=envKwargs)
env = gym.make('RPiLEDEnv-v0', resizeCamImagePct=50, ledHSVLower=np.array([0, 0, 252]), ledHSVHigher=np.array([31, 9, 255]), rPiIP='192.168.0.183', rPiPort=50000)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo2_rpi_led")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='console')
