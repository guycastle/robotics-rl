import gym
import envs
import numpy as np
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import warnings
warnings.filterwarnings('ignore')

env = gym.make('RPiLEDEnv-v0', resizeCamImagePct=50, ledHSVLower=np.array([0, 0, 252]),
               ledHSVHigher=np.array([31, 9, 255]), rPiIP='192.168.0.183', rPiPort=50000, episodLength=100, bullseye=15)

model = PPO2(MlpPolicy, env, verbose=1)
model.load("ppo2_rpi_led.zip")
model.learn(total_timesteps=50000)
model.save("ppo2_rpi_led")

