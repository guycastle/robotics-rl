import gym
import envs
import numpy as np
from stable_baselines.deepq.policies import CnnPolicy, MlpPolicy
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, CallbackList
from stable_baselines import DQN
import warnings
warnings.filterwarnings('ignore')

env = gym.make(
    'RPiLEDEnv-v0', resizeCamImagePct=50, ledHSVLower=np.array([0, 0, 252]), ledHSVHigher=np.array([31, 9, 255]),
    rPiIP='192.168.0.183', rPiPort=50000, episodeLength=100, bullseye=10
)

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-20, verbose=1)

eval_callback = EvalCallback(env, best_model_save_path='./logs/best',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False, callback_on_new_best=callback_on_best)

# Added checkpoint because I lost model data after a crash when the webcam shutdown because the screen went to sleep :(
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='ppo1_model')

cb = CallbackList([eval_callback, checkpoint_callback])

model = DQN(MlpPolicy, env, verbose=1, double_q=True, tensorboard_log='./logs/')
model.learn(total_timesteps=2000, callback=cb)
model.save("dqn_rpi_led")
