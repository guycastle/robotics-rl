import cv2
import time
import numpy as np

cam = cv2.VideoCapture(0)
resizePct = 50

def detect_red_from_webcam():
    _, img = cam.read()
    w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(w, h)
    # In case webcam doesn't return any images, sleep
    if img.any() is None:
        time.sleep(0.01)
    # Resize the percent
    img = cv2.resize(img, (int(img.shape[1] * resizePct / 100), int(img.shape[0] * resizePct / 100)))
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Pixel color (values obtained with util.HSVSliderCountours.py)
    low_red = np.array([0, 0, 252])
    high_red = np.array([31, 9, 255])
    mask = cv2.inRange(hsv_frame, low_red, high_red)
    # Perform a 3x3 dilation to fill the gaps in the LED
    mask2 = mask.astype(np.uint8)
    mask2 = cv2.dilate(mask2, np.ones((3, 3)))
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask2)


    coX, coY = -1, -1
    max_area = None

    for stat, center in zip(stats[1:], centroids[1:]):
        area = stat[4]

        if (max_area is None) or (area > max_area):
            x, y = center
            max_area = area

    # Filtering the mask for noise
    # kernel_open = np.ones((4, 4))
    # kernel_close = np.ones((40, 40))
    # red_open = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel_open)
    # red_close = cv2.morphologyEx(red_open, cv2.MORPH_CLOSE, kernel_close)

    # Find a squarish countour :)
    # frame_gray = cv2.cvtColor(red_close, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(mask2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coX = 0
    coY = 0
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if (w < h * 1.15 and w > h * 0.85):
            coX = x
            coY = y
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # This also gives the center of the red dot
    center = (coX, coY)
    cv2.imshow("Original", img)
    cv2.waitKey(1)

while True:
    detect_red_from_webcam()