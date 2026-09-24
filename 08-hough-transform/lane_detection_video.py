import cv2
import numpy as np

vid = cv2.VideoCapture("media/highway_lanes.mp4")

# Only the yellow lane markings matter, so they are isolated in HSV
# before edge detection instead of running Canny on the whole frame.
lower_yellow = np.array([18, 94, 140], np.uint8)
upper_yellow = np.array([48, 255, 255], np.uint8)

while True:
    ret, frame = vid.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    edges = cv2.Canny(mask, 75, 250)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)

    if lines is not None:
        for line in lines:
            (x1, y1, x2, y2) = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow("Mask", mask)
    cv2.imshow("Edges", edges)
    cv2.imshow("Lanes", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
