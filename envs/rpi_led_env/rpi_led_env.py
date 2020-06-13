import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import struct
import socket

class RPiLEDEnv(gym.Env):

    def __init__(self):
        print('Environment initialized')
    def step(self):
        print('Step successful!')
    def reset(self):
        print('Environment reset')