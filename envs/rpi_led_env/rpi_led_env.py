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
        4: 'n',  # Nothing/Idle
        5: 'ul',  # Up & Left
        6: 'ur',  # Up & Right
        7: 'dl',  # Down & Left
        8: 'dr'  # Down & Right
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
        # Let's try normalizing the positional data with the center being [0,0]
        # self.center = np.array([0,0])


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
            high=np.array([
                #  (self.resolution[0] / 100 * resizeCamImagePct), (self.resolution[1] / 100 * resizeCamImagePct),
                # If we normalize, it's also 1
                1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0
            ]),
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
        self.reward = self.calcReward(ledPos[0], ledPos[1])
        # Normalize the coordinates after having calculated the reward
        ledPos = [self.normalizeCoords(ledPos[0], self.resolution[0]),
                  self.normalizeCoords(ledPos[1], self.resolution[1])]
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
        # Additional reward based on percentage of distance from center:
        bullseye = self.calculateDiagonalOfSquare(self.center[0], self.center[1]) / 100 * self.bullseye
        if bullseye > distance > 10:
            distance -= (bullseye / 2)
            if distance >= 0:
                # Don't give a positive reward
                distance = -1
            print('Hit bullseye ({bullseye}): {reward}'.format(bullseye=bullseye, reward=distance))
        return -distance

    def calculateDiagonalOfSquare(self, width, height):
        return (width ** 2 + height ** 2) ** 0.5

    def detectLED(self):
        _, img = self.camera.read()
        # In case webcam doesn't return any images, sleep and try again
        if img is None:
            time.sleep(0.01)
            return self.detectLED()
        else:
            # Resize the percent
            img = cv2.resize(img, (
                int(img.shape[1] * self.resizeCamImagePct / 100), int(img.shape[0] * self.resizeCamImagePct / 100)))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Pixel color (values obtained with util.HSVSliderContours.py)
            lowHsv = self.ledHSVLower
            highHsv = self.ledHSVHigher
            mask = cv2.inRange(hsv, lowHsv, highHsv)
            mask2 = mask.astype(np.uint8)
            mask2 = cv2.dilate(mask2, np.ones((3, 3)))

            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask2)

            coX, coY = -1.0, -1.0
            max_area = None

            for stat, center in zip(stats[1:], centroids[1:]):
                area = stat[4]

                if (max_area is None) or (area > max_area):
                    coX, coY = center
                    max_area = area

            img2 = np.copy(img)
            coX, coY = int(coX), int(coY)

            img2[coY - 10:coY + 10, coX - 10:coX + 10, :] = (100, 100, 255)

            img2[self.center[1] - 5:self.center[1] +5, self.center[0] - 5:self.center[0] + 5, :] = (255, 100, 255)
            cv2.imshow("Original", img2)
            cv2.waitKey(1)
            # If you don't detect anything, try again
            if coX == -1.0 or coY == -1.0:
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

