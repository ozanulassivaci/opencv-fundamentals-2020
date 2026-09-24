import cv2

img1 = cv2.imread("media/bitwise_1.png")
img2 = cv2.imread("media/bitwise_2.png")

bit_or = cv2.bitwise_or(img2, img1)

cv2.imshow("Bitwise OR", bit_or)
cv2.imshow("Original 1", img1)
cv2.imshow("Original 2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
