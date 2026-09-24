import cv2

img = cv2.imread("media/portrait_1.jpg", cv2.IMREAD_GRAYSCALE)

# WINDOW_NORMAL makes the window resizable instead of fixed to the image size.
cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
cv2.imshow("Grayscale", img)
cv2.imwrite("media/portrait_1_gray.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
