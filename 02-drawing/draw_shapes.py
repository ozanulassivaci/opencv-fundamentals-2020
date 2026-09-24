import cv2
import numpy as np

# np.zeros gives a black canvas; adding 255 turns every channel white.
canvas = np.zeros((512, 512, 3), dtype=np.uint8) + 255

cv2.line(canvas, (50, 50), (512, 512), (255, 0, 0), thickness=5)
cv2.line(canvas, (512, 1), (50, 50), (255, 0, 0), thickness=5)
cv2.rectangle(canvas, (50, 50), (150, 150), (0, 255, 0), 10)
cv2.circle(canvas, (250, 250), 200, (0, 0, 255), 5)

# Triangle built from three separate line segments
p1 = (100, 200)
p2 = (50, 50)
p3 = (300, 100)

cv2.line(canvas, p1, p2, (0, 0, 0), 4)
cv2.line(canvas, p2, p3, (0, 0, 0), 4)
cv2.line(canvas, p1, p3, (0, 0, 0), 4)

# polylines draws the same kind of shape from a list of vertices in one call
points = np.array([[110, 400], [330, 400], [290, 460], [100, 480]], np.int32)
cv2.polylines(canvas, [points], True, (0, 0, 100), 5)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
