import cv2

img = cv2.imread("media/portrait_blue.jpg")

# OpenCV loads images as BGR; showing the converted arrays makes the channel order visible.
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("BGR", img)
cv2.imshow("RGB", rgb)
cv2.imshow("HSV", hsv)
cv2.imshow("Gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
