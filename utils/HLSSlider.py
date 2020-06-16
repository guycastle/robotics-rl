import cv2
import numpy as np

def nothing(x):
    pass

# Load image
_, image = cv2.VideoCapture(0).read()

# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('LMin', 'image', 0, 255, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('LMax', 'image', 0, 255, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('LMax', 'image', 255)
cv2.setTrackbarPos('SMax', 'image', 255)

# Initialize HSV min/max values
hMin = sMin = lMin = hMax = sMax = lMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    lMin = cv2.getTrackbarPos('LMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    lMax = cv2.getTrackbarPos('LMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, lMin])
    upper = np.array([hMax, sMax, lMax])

    # Convert to HSV format and color threshold
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)
    mask = cv2.inRange(hls, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != lMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != lMax)):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , lMin, hMax, sMax , lMax))
        phMin = hMin
        psMin = sMin
        pvMin = lMin
        phMax = hMax
        psMax = sMax
        pvMax = lMax

    # Display result image
    cv2.imshow('image', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()