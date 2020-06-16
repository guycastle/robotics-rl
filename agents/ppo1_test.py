import gym
import envs
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines import PPO1
import warnings
warnings.filterwarnings('ignore')

env = gym.make(
    'RPiLEDEnv-v0', resizeCamImagePct=50, ledHSVLower=np.array([0, 0, 252]), ledHSVHigher=np.array([31, 9, 255]),
    rPiIP='192.168.0.183', rPiPort=50000, episodeLength=100, bullseye=10
)

model = PPO1(MlpPolicy, env, verbose=1, tensorboard_log="./logs/")

model.load("/Users/guillaumevandecasteele/PycharmProjects/robotics/logs/New Folder With Items/ppo1_model_14848_steps.zip")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()