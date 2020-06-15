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

    def __init__(self, resizeCamImagePct, ledHSVLower, ledHSVHigher, rPiIP, rPiPort, episodeLength, bullseye):
        super(RPiLEDEnv, self).__init__()
        # Initialize the webcam
        self.camera = cv2.VideoCapture(0)
        # Store the webcam resolution
        self.resolution = np.array([self.camera.get(cv2.CAP_PROP_FRAME_WIDTH) / 100 * resizeCamImagePct,
                                    self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT) / 100 * resizeCamImagePct])
        # Store the value by which to resize the webcam image
        self.resizeCamImagePct = resizeCamImagePct
        # Calculate the center of the image
        self.center = np.array([int(self.resolution[0] / 2),
                                int(self.resolution[1] / 2)])
        # Set the lower and higher bounds for the LED detection
        self.ledHSVLower = ledHSVLower
        self.ledHSVHigher = ledHSVHigher
        # Open a socket to the Raspberry Pi
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.socket.connect((rPiIP, rPiPort))
        # Store the bullseye value, this is the percentage of distance from the center to give additional reward
        self.bullseye = bullseye
        # Make the episode last X times before returning True
        self.episodeStep = 0
        self.episodeLength = episodeLength
        # Define the action space, based on the action map defined above
        self.action_space = spaces.Discrete(len(self.ACTION_MAP))
        # Define the observation space. It's an unbounded array of 8 values, being
        # 1. x-coordinate of LED. normalized to [-1,+1]
        # 2. y-coordinate of LED. normalized to [-1,+1]
        # 3. x-value of Pi gyroscope. lower bound -1, higher bound +1
        # 4. y-value of Pi gyroscope. lower bound -1, higher bound +1
        # 5. z-value of Pi gyroscope. lower bound -1, higher bound +1
        # 6. x-value of Pi accelerometer. lower bound -1, higher bound +1
        # 7. y-value of Pi accelerometer. lower bound -1, higher bound +1
        # 8. z-value of Pi accelerometer. lower bound -1, higher bound +1
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array(
                [(self.resolution[0] / 100 * resizeCamImagePct), (self.resolution[1] / 100 * resizeCamImagePct), 1.0,
                 1.0, 1.0,
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
        # Normalize the coordinates
        # ledPos = [self.normalizeCoords(ledPos[0], self.resolution[0]),
        #           self.normalizeCoords(ledPos[1], self.resolution[1])]
        self.reward = self.calcReward(ledPos[0], ledPos[1])
        self.state = np.array(ledPos + piData)
        self.episodeStep += 1
        return self.state, self.reward, self.episodeStep >= self.episodeLength, {}

    def reset(self):
        self.send('n')
        self.state = np.array(self.detectLED() + self.rcv())
        self.episodeStep = 0
        return self.state

    def render(self, mode='human'):
        s = self.state
        print(self.resolution)
        print(self.center)
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
        distance = self.calculateDiagonalOfSquare(w, h)
        print('unaltered distance: {dist}'.format(dist=distance))
        # Additional reward based on percentage of distance from center:
        bullseye = self.calculateDiagonalOfSquare(self.center[0], self.center[1]) / 100 * self.bullseye
        if distance < bullseye:
            distance -= bullseye
            print('Hit bullseye ({bullseye}): {reward}'.format(bullseye=bullseye, reward=distance))
        return -distance

    def calculateDiagonalOfSquare(self, width, height):
        return (width ** 2 + height ** 2) ** 0.5

    def detectLED(self):
        _, img = self.camera.read()
        # In case webcam doesn't return any images, sleep
        if img is None:
            time.sleep(0.01)
        # Resize the percent
        img = cv2.resize(img, (
        int(img.shape[1] * self.resizeCamImagePct / 100), int(img.shape[0] * self.resizeCamImagePct / 100)))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Pixel color (values obtained with util.HSVSliderContours.py)
        lowHsv = self.ledHSVLower
        highHsv = self.ledHSVHigher
        mask = cv2.inRange(hsv, lowHsv, highHsv)
        maskedImg = cv2.bitwise_and(img, img, mask=mask)

        # Filtering the mask for noise
        kernelOpen = np.ones((4, 4))
        kernelClose = np.ones((40, 40))
        openMorph = cv2.morphologyEx(maskedImg, cv2.MORPH_OPEN, kernelOpen)
        closeMorph = cv2.morphologyEx(openMorph, cv2.MORPH_CLOSE, kernelClose)

        # Find a squarish contour :)
        gray = cv2.cvtColor(closeMorph, cv2.COLOR_BGR2GRAY)
        threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Set the coords to negative value; this should indicate that the LED is not visible onscreen
        coX = -1.0
        coY = -1.0
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if (w < h * 1.15 and w > h * 0.85):
                coX = x + (w / 2)
                coY = y + (h / 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.imshow("Original", img)
        cv2.waitKey(1)
        # If you don't detect anything, try again
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

    # Normalize the coordinates/resolution to a [-1,+1] space analogous to the pi sensor data
    def normalizeCoords(self, c, maximum):
        return (2 * (c / maximum)) - 1

