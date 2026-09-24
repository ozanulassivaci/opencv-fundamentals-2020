import cv2
import numpy as np

circle = np.zeros((512, 512, 3), np.uint8) + 255
cv2.circle(circle, (256, 256), 60, (255, 0, 0), -1)

rectangle = np.zeros((512, 512, 3), np.uint8) + 255
cv2.rectangle(rectangle, (150, 150), (350, 350), (0, 0, 255), -1)

# dst = circle * 0.5 + rectangle * 0.5 + 0, which blends the two images evenly.
blended = cv2.addWeighted(circle, 0.5, rectangle, 0.5, 0)

cv2.imshow("Circle", circle)
cv2.imshow("Rectangle", rectangle)
cv2.imshow("Blended", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
