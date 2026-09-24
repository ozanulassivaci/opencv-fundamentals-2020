import cv2
import numpy as np

cap = cv2.VideoCapture("media/dog.mp4")

# White has low saturation and high value in HSV regardless of hue,
# so the hue range is left fully open.
sensitivity = 15
lower_white = np.array([0, 0, 255 - sensitivity])
upper_white = np.array([255, sensitivity, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_white, upper_white)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Esc
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
