import cv2
import time
import numpy as np

# Inspired by https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv

cam = cv2.VideoCapture(0)
resizePct = 50

def nothing(x):
    pass

cv2.namedWindow('Original')

cv2.createTrackbar('HMin', 'Original', 0, 179, nothing)
cv2.createTrackbar('SMin', 'Original', 0, 255, nothing)
cv2.createTrackbar('VMin', 'Original', 0, 255, nothing)
cv2.createTrackbar('HMax', 'Original', 0, 179, nothing)
cv2.createTrackbar('SMax', 'Original', 0, 255, nothing)
cv2.createTrackbar('VMax', 'Original', 0, 255, nothing)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', 'Original', 179)
cv2.setTrackbarPos('SMax', 'Original', 255)
cv2.setTrackbarPos('VMax', 'Original', 255)

while True:
    _, frame = cam.read()
    if frame.any() is None:
        time.sleep(0.001)
        continue
    else:
        frame = cv2.resize(frame, (int(frame.shape[1] * resizePct / 100), int(frame.shape[0] * resizePct / 100)))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hMin = cv2.getTrackbarPos('HMin', 'Original')
        sMin = cv2.getTrackbarPos('SMin', 'Original')
        vMin = cv2.getTrackbarPos('VMin', 'Original')
        hMax = cv2.getTrackbarPos('HMax', 'Original')
        sMax = cv2.getTrackbarPos('SMax', 'Original')
        vMax = cv2.getTrackbarPos('VMax', 'Original')

        # Set minimum and maximum HSV values to display
        low_red = np.array([hMin, sMin, vMin])
        high_red = np.array([hMax, sMax, vMax])

        # Red color
        # low_red = np.array([0, 0, 0])
        # high_red = np.array([1, 1, 255])
        red_mask = cv2.inRange(hsv_frame, low_red, high_red)
        red = cv2.bitwise_and(frame, frame, mask=red_mask)

        # Filtering the mask for noise
        kernel_open = np.ones((4, 4))
        kernel_close = np.ones((40, 40))
        red_open = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel_open)
        red_close = cv2.morphologyEx(red_open, cv2.MORPH_CLOSE, kernel_close)

        # Finding Contours and draw them
        frame_gray = cv2.cvtColor(red_close, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Print if there is a change in HSV value
        if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (
                pvMax != vMax)):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
                hMin, sMin, vMin, hMax, sMax, vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Show live video's of the different masks
        cv2.imshow("Original", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
