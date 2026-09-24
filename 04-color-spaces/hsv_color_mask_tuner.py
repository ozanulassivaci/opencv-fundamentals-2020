import cv2
import numpy as np


def on_trackbar_change(value):
    # Values are polled every frame with getTrackbarPos, so the callback has nothing to do.
    pass


cap = cv2.VideoCapture(0)
cv2.namedWindow("Settings")

# OpenCV stores hue as 0-180 (degrees / 2), saturation and value as 0-255.
cv2.createTrackbar("Lower-Hue", "Settings", 0, 180, on_trackbar_change)
cv2.createTrackbar("Lower-Saturation", "Settings", 0, 255, on_trackbar_change)
cv2.createTrackbar("Lower-Value", "Settings", 0, 255, on_trackbar_change)
cv2.createTrackbar("Upper-Hue", "Settings", 180, 180, on_trackbar_change)
cv2.createTrackbar("Upper-Saturation", "Settings", 255, 255, on_trackbar_change)
cv2.createTrackbar("Upper-Value", "Settings", 255, 255, on_trackbar_change)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lh = cv2.getTrackbarPos("Lower-Hue", "Settings")
    ls = cv2.getTrackbarPos("Lower-Saturation", "Settings")
    lv = cv2.getTrackbarPos("Lower-Value", "Settings")
    uh = cv2.getTrackbarPos("Upper-Hue", "Settings")
    us = cv2.getTrackbarPos("Upper-Saturation", "Settings")
    uv = cv2.getTrackbarPos("Upper-Value", "Settings")

    lower_color = np.array([lh, ls, lv])
    upper_color = np.array([uh, us, uv])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(3) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
