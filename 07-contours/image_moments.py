import cv2

img = cv2.imread("media/contour.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# The centroid of a shape is (m10 / m00, m01 / m00).
moments = cv2.moments(thresh)
cx = int(moments["m10"] / moments["m00"])
cy = int(moments["m01"] / moments["m00"])

cv2.circle(img, (cx, cy), 5, (255, 255, 0), -1)

cv2.imshow("Centroid", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
