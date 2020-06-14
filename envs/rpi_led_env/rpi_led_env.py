import gym
import numpy as np
import cv2
import struct
import pickle
import socket
import time
from gym import spaces

import warnings
warnings.filterwarnings('ignore')


class RPiLEDEnv(gym.Env):
    metadata = {'render.modes': ['console', 'human']}

    RENDER_FORMAT = "Reward: {reward}, x: {x}, y: {y}, gyroX: {gyroX}, gyroY: {gyroY}, gyroZ: {gyroZ}, accelX: {accelX}, accelY: {accelY}, accelZ: {accelZ}"

    ACTION_MAP = {
        0: 'u',  # Up
        1: 'd',  # Down
        2: 'l',  # Left
        3: 'r',  # Right
        4: 'ul',  # Up & Left
        5: 'ur',  # Up & Right
        6: 'dl',  # Down & Left
        7: 'dr',  # Down & Right
        8: 'n'  # Nothing/Idle
    }

    def __init__(self, resizeCamImagePct, ledHSVLower, ledHSVHigher, rPiIP, rPiPort):
        super(RPiLEDEnv, self).__init__()
        # Initialize the webcam
        self.camera = cv2.VideoCapture(0)
        # Store the native webcam resolution
        self.resolution = [self.camera.get(cv2.CAP_PROP_FRAME_WIDTH), self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)]
        # Store the value by which to resize the webcam image
        self.resizeCamImagePct = resizeCamImagePct
        # Calculate the center of the image
        self.center = np.array([int(self.resolution[0] / 2 / 100 * resizeCamImagePct),
                                int(self.resolution[1] / 2 / 100 * resizeCamImagePct)])
        # Set the lower and higher bounds for the LED detection
        self.ledHSVLower = ledHSVLower
        self.ledHSVHigher = ledHSVHigher
        # Open a socket to the Raspberry Pi
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.socket.connect((rPiIP, rPiPort))
        # Define the action space, based on the action map defined above
        self.action_space = spaces.Discrete(len(self.ACTION_MAP))
        # Define the observation space. It's an unbounded array of 8 values, being
        # 1. x-coordinate of LED. lower bound -1 (not visible in image), higher bound the width of the image
        # 2. y-coordinate of LED. lower bound -1 (not visible in image), higher bound the height of the image
        # 3. x-value of Pi gyroscope. lower bound -1, higher bound +1
        # 4. y-value of Pi gyroscope. lower bound -1, higher bound +1
        # 5. z-value of Pi gyroscope. lower bound -1, higher bound +1
        # 6. x-value of Pi accelerometer. lower bound -1, higher bound +1
        # 7. y-value of Pi accelerometer. lower bound -1, higher bound +1
        # 8. z-value of Pi accelerometer. lower bound -1, higher bound +1
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array(
                [(self.resolution[0] / 100 * resizeCamImagePct), (self.resolution[1] / 100 * resizeCamImagePct), 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0]),
            shape=(8,),
            dtype=np.float32)
        # Initial state (off-screen and gyroscope and accelerometer set to zero
        self.state = np.array([-1.0, -1.0, 0, 0, 0, 0, 0, 0])
        # Initial reward
        self.reward = self.calcReward(0, 0)
        self.reset()

    # Basic inspiration taken from https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # Execute the action
        self.send(self.ACTION_MAP[action])
        # Receive the gyro/accelerometer data
        piData = self.rcv()
        # Get the new position of the LED pixel
        ledPos = self.detectLED()
        self.state = np.array(ledPos + piData)
        # If the camera is off-screen, return a 0 reward
        if self.state[0] == -1:
            self.reward = 0
        else:
            self.reward = self.calcReward(ledPos[0], ledPos[1])
        return self.state, self.reward, True, {}

    def reset(self):
        self.send('n')
        self.state = np.array(self.detectLED() + self.rcv())
        return self.state

    def render(self, mode='human'):
        s = self.state
        print(self.state.shape)
        print(s[0])
        if mode == 'human':
            # TODO make it display the detected LED with contours in a window.
            # cv2.rectangle(self.img, (int(s[0]), int(s[1])), (int(s[0]) + 4, int(s[0]) + 4), (255, 0, 255), 2)
            # cv2.imshow("Capture", self.img)
            # cv2.waitKey(1)
            print(
                self.RENDER_FORMAT.format(reward=self.reward, x=s[0], y=s[1], gyroX=s[2], gyroY=s[3], gyroZ=s[4],
                                          accelX=s[5], accelY=s[6],
                                          accelZ=s[7]))
        else:
            print(
                self.RENDER_FORMAT.format(reward=self.reward, x=s[0], y=s[1], gyroX=s[2], gyroY=s[3], gyroZ=s[4],
                                          accelX=s[5], accelY=s[6],
                                          accelZ=s[7]))

    # calculate the reward based on the detected coordinates
    def calcReward(self, x, y):
        # The reward is the negative value of the distance between the center of the image & the detected LED
        w = abs(self.center[0] - x)
        h = abs(self.center[1] - y)
        # The distance is the square root of the width^2 + length^2
        distance = (w ** 2 + h ** 2) ** 0.5
        return -distance

    def detectLED(self):
        _, img = self.camera.read()
        # In case webcam doesn't return any images, sleep
        if img.any() is None:
            time.sleep(0.01)
        # Resize the percent
        img = cv2.resize(img, (
        int(img.shape[1] * self.resizeCamImagePct / 100), int(img.shape[0] * self.resizeCamImagePct / 100)))
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Pixel color (values obtained with util.HSVSliderContours.py)
        low_red = self.ledHSVLower
        high_red = self.ledHSVHigher
        red_mask = cv2.inRange(hsv_frame, low_red, high_red)
        red = cv2.bitwise_and(img, img, mask=red_mask)

        # Filtering the mask for noise
        kernel_open = np.ones((4, 4))
        kernel_close = np.ones((40, 40))
        red_open = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel_open)
        red_close = cv2.morphologyEx(red_open, cv2.MORPH_CLOSE, kernel_close)

        # Find a squarish contour :)
        frame_gray = cv2.cvtColor(red_close, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Set the coords to negative value; this should indicate that the LED is not visible onscreen
        coX = -1.0
        coY = -1.0
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if (w < h * 1.15 and w > h * 0.85):
                coX = x + (w / 2)
                coY = y + (h / 2)
        if coX == -1 or coY == -1:
            return self.detectLED()
        else:
            return [coX, coY]

    def rcv(self):
        data = self.socket.recv(4, socket.MSG_WAITALL)
        data_len = struct.unpack('>i', data)[0]
        data = self.socket.recv(data_len, socket.MSG_WAITALL)
        return pickle.loads(data)

    def send(self, data):
        data = pickle.dumps(data)
        self.socket.sendall(struct.pack('>i', len(data)))
        self.socket.sendall(data)
