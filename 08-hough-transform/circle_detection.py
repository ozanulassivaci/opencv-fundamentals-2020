import cv2
import numpy as np


def detect_circles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # HoughCircles is very sensitive to noise; the median blur removes most false circles.
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, img.shape[0] / 64,
                               param1=200, param2=10, minRadius=5, maxRadius=30)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
    return img


coins = cv2.imread("media/coins.jpg")
balls = cv2.imread("media/balls.jpg")

cv2.imshow("Coins", detect_circles(coins))
cv2.imshow("Balls", detect_circles(balls))
cv2.waitKey(0)
cv2.destroyAllWindows()
