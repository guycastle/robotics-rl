import socket
import pickle
import struct
from sense_hat import SenseHat

red = (255, 0, 0)

# Use to control Sense HAT
sense = SenseHat()
# Reinitialize the LED matrix and light up the middle pixel
sense.clear()
sense.set_pixel(3, 3, red)

# Create server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', 50000))
s.listen(10)
# Accept a client
conn, addr = s.accept()


def recv(s):
    data = s.recv(4, socket.MSG_WAITALL)
    data_len = struct.unpack('>i', data)[0]
    data = s.recv(data_len, socket.MSG_WAITALL)
    return pickle.loads(data)


def send(s, data):
    data = pickle.dumps(data)
    s.sendall(struct.pack('>i', len(data)))
    s.sendall(data)


# Decode the action sent by the environment
def decodeAndPerformAction(action):
    # Search which pixel is lit up. For some reason, the value isn't 255 as set when lighting, but 248 instead ...
    lit_pixel = sense.get_pixels().index([248, 0, 0])
    # Get the X coordinate by getting the modulus 8 of the index
    x = lit_pixel % 8
    # Get the Y coordinate by getting the integer value of the index divided by
    y = int(lit_pixel / 8)

    if action == 'u':
        y -= 1
    elif action == 'd':
        y += 1
    elif action == 'l':
        x -= 1
    elif action == 'r':
        x += 1
    # Extra actions, diagonally
    elif action == 'ul':
        y -= 1
        x -= 1
    elif action == 'ur':
        y -= 1
        x += 1
    elif action == 'dl':
        x -= 1
        y += 1
    elif action == 'dr':
        x += 1
        y += 1
    # Do nothing action, leave x and y as is.
    print('Action: {action}, X: {x}, Y: {y}'.format(action=action, x=x, y=y))
    # Clear the LED matrix before updating it
    sense.clear()
    sense.set_pixel(x, y, red)


while True:
    # Receive the action from the environment and perform it
    decodeAndPerformAction(recv(conn))
    # Return information (accelerometer, gyroscope, ...): https://pythonhosted.org/sense-hat/api/#imu-sensor
    gyro = sense.get_gyroscope_raw()
    accel = sense.get_accelerometer_raw()
    send(conn, [gyro['x'], gyro['y'], gyro['z'], accel['x'], accel['y'], accel['z']])
