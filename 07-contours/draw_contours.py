import cv2

img = cv2.imread("media/contours_nested.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Index 1 draws a single contour; -1 would draw all of them.
cv2.drawContours(img, contours, 1, (0, 0, 255), 3)

cv2.imshow("Contour", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
