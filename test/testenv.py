import gym
import envs
import numpy as np
import warnings
warnings.filterwarnings('ignore')

env = gym.make(
    'RPiLEDEnv-v0', resizeCamImagePct=50, ledHSVLower=np.array([0, 0, 252]), ledHSVHigher=np.array([31, 9, 255]),
    rPiIP='192.168.0.183', rPiPort=50000, episodeLength=100, bullseye=15
)

obs = env.reset()
while True:
    obs, rewards, dones, info = env.step(env.action_space.sample())
    env.render()
