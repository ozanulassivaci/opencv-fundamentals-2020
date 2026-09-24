import cv2
import numpy as np

# A 10x10 black image with a few pixels set by hand (values are BGR, not RGB).
img = np.zeros((10, 10, 3), np.uint8)
img[0, 0] = (255, 255, 255)
img[0, 1] = (255, 200, 255)
img[0, 2] = (150, 255, 255)
img[0, 3] = (255, 255, 15)

# Upscale so the individual pixels are large enough to see.
img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
cv2.imshow("Pixels", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
