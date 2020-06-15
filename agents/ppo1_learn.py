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
    rPiIP='192.168.0.183', rPiPort=50000, episodeLength=100, bullseye=15
)

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-20, verbose=1)

eval_callback = EvalCallback(env, best_model_save_path='./logs/best',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False, callback_on_new_best=callback_on_best)

model = PPO1(MlpPolicy, env, verbose=1, tensorboard_log="./logs/")
model.save("ppo1_rpi_led.zip")
model.learn(total_timesteps=60000)
model.save("ppo1_rpi_led")
