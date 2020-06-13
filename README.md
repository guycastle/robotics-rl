# Project – AI Robotics class

This project builds on the Raspberry Pi and illustrates several aspects of real-world robotics:

1. Hardware sensors (orientation, acceleration)
2. Image processing and its challenges
3. The functionality to be learned by the agent is extremely easy to express (and hence to reward), but would be almost impossible to implement by hand.

The last point is particularly important, as Reinforcement Learning is dedicated to tasks that cannot be solved otherwise. It is pointless to apply Reinforcement Learning for the sake of doing it, to a task that can be solved with other methods. Reinforcement Learning both is the last-resort solution, and aims squarely at super-human performance (that is, producing policies that are better than what any other method would have provided).

## Project Goal
The project is easy to define, as most of its challenges are in the technicalities and learning, not in the objective.

The Raspberry Pi with its Sense Hat has sensors and an LED matrix. The objective of the project is to allow a person to hold his/her Raspberry Pi in front of a webcam (the webcam of a laptop will do). The LED matrix displays a single red dot. The idea is that the red dot must be at the center of the image captured by the webcam, even if the person holding the Raspberry Pi shakes (I don’t havea Sense Hat, so I used a piece of paper to mimic the red LED)

This means that as the person shakes the Raspberry Pi, the red dot on the LED matrix has to move accordingly so that it remains at the center of what the webcam sees. This has to work even if the Raspberry Pi is not fully vertical. If it is impossible to center the dot, because the Raspberry Pi is too off-center for its LED matrix to cover the center of the image, the red dot must be as close to thecenter of the image as possible.

This project may seem weird, but it is a the center of image stabilization on photo cameras and medical devices. The task to be performed, inverting the shaking of the user, is almost impossible todesign by a human, but can be learned by a Reinforcement Learning agent. It is therefore a real-world application of Reinforcement Learning.

## Detailed Steps
The project is successful as soon as the red dot remains in the center of the image. Because this project requires many components to interact, here is a roughly-ordered TODO list:

1. Implement a Python socket server on the Raspberry Pi, to allow the Raspberry Pi to send sensor readings to a computer on the same (Wifi) network, and obtain from the computer target X-Y coordinates for the red dot on the LED matrix. Code is provided at the end of this assignment, as this is a Robotics class, not a Network programming class.
2. Implement on the computer a program that tells the Raspberry Pi to put the red dot somewhere. On the Raspberry Pi, use the SenseHat Python library to put the red dot where the computer wants.
3. Implement on the computer a Python with OpenCV program that accesses the webcam, and does image processing on it to detect where the red dot is (see first exercise session on OpenCV, for detecting red objects).
4. On the Raspberry Pi, use the SenseHat library to read accelerometer and gyroscopic data from the SenseHat sensors. Allow the computer to obtain these readings on the network.
5. On the computer, implement an OpenAI Gym environment that provides as observations the following: accelerometer and gyroscopic data (6 floats), and the current X-Y location of the red dot in the image observed by the webcam (2 floats). The environment implements 4 actions: move the red dot up/down/left/right in the LED matrix of the Raspberry Pi.
6. Implement a reward function, that basically rewards the agent when the red dot is near the center of the image observed by the webcam. If you can compute the distance between the red dot and the center (in pixels, or percentages of the size of the image, it does not matter), you can provide -distance as a reward to the agent. This will punish it more when the dot is further from the center of the image, less when the dot is near the center. Because the agent wants to maximize its cumulative reward, it will try to keep the red dot as close to the centeras possible.
7. Evaluate several Reinforcement Learning agents on the OpenAI Gym environment you have designed. Most RL agents available on Github are compatible with the Gym, so evaluating a bunch of them is easy. I recommend PPO in “stable-baselines”, and BDPI, developed at the AI lab. The lecture on RL gives more details. These agents need their neural networks to be configured. I believe that a single hidden layer of 128 neurons should be enough.
### Network Code
On the Raspberry Pi, you can create a socket server with the following code:
```python
import socket
# Create server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', 9395))
s.listen(10)
while True:    
   # Accept a client    
   conn, addr = s.accept()    
   # conn is now a “socket”, if you write to it, the client receives that data.    
   # If you read from it, you get what the client sent you
```
On the computer, you can connect to the Raspberry Pi socket server with:

```python
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.connect(('IP address of the raspberry pi', 9395))
# s is now a “socket”
```

On both ends (computer and Raspberry Pi), you can now read and write data to/from the socket using those two functions. They take as parameters Python objects, which means that you can send anything you want on the socket: strings, integers, tuples, Numpy arrays, images, dictionaries, etc. Python will serialize and deserialize everything for you:

```python
import pickle
import struct

def send(s, data):    
    data = pickle.dumps(data)    
    s.sendall(struct.pack('>i', len(data)))    
    s.sendall(data)

def recv(s):    
    data = s.recv(4, socket.MSG_WAITALL)    
    data_len = struct.unpack('>i', data)[0]    
    data = s.recv(data_len, socket.MSG_WAITALL)
    return pickle.loads(data)
```