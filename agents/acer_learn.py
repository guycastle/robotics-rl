import gym
import envs
from envs.rpi_led_env.rpi_led_env import RPiLEDEnv
import numpy as np
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import ACER
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, CallbackList
import warnings
warnings.filterwarnings('ignore')

envArgsDict = {
    'resizeCamImagePct': 50, 'ledHSVLower': np.array([0, 0, 252]), 'ledHSVHigher':np.array([31, 9, 255]),
    'rPiIP': '192.168.0.183', 'rPiPort':50000, 'episodeLength':100, 'bullseye':10
}

env = make_vec_env(RPiLEDEnv, env_kwargs=envArgsDict)


callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-20, verbose=1)

eval_callback = EvalCallback(env, best_model_save_path='./logs/best',
                             log_path='./logs/', eval_freq=5000,
                             deterministic=True, render=False, callback_on_new_best=callback_on_best)

# Added checkpoint because I lost model data after a crash when the webcam shutdown because the screen went to sleep :(
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='ppo1_model')

cb = CallbackList([checkpoint_callback, eval_callback])

policy_kwargs = {'layers':[128]}

model = ACER(MlpLnLstmPolicy, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log='./logs/')
model.learn(total_timesteps=10000, callback=cb)
model.save('acer_rpi_lid')
print('model saved')
