import cv2
import numpy as np

canvas = np.zeros((512, 512, 3), dtype=np.uint8) + 255

fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]

for i, font in enumerate(fonts):
    cv2.putText(canvas, "Test", (20, 150 + i * 150), font, 4, (0, 0, 0), 2, cv2.LINE_AA)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
