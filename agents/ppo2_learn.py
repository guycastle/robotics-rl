import gym
import envs
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, CallbackList
# from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import warnings
warnings.filterwarnings('ignore')

env = gym.make('RPiLEDEnv-v0', resizeCamImagePct=50, ledHSVLower=np.array([0, 0, 252]),
               ledHSVHigher=np.array([31, 9, 255]), rPiIP='192.168.0.183', rPiPort=50000, episodLength=100, bullseye=5)

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-20, verbose=1)

eval_callback = EvalCallback(env, best_model_save_path='./logs/best',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False, callback_on_new_best=callback_on_best)

# Added checkpoint because I lost model data after a crash when the webcam shutdown because the screen went to sleep :(
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='ppo1_model')

cb = CallbackList([checkpoint_callback, eval_callback])

model = PPO2(MlpPolicy, env, verbose=1)
#model.load("ppo2_rpi_led.zip")
model.learn(total_timesteps=50000, callback=cb)
model.save("ppo2_rpi_led")

