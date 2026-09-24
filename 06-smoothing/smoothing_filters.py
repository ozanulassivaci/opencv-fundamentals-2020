import cv2

img_filter = cv2.imread("media/noisy_filter.png")
img_median = cv2.imread("media/salt_and_pepper_noise.png")
img_bilateral = cv2.imread("media/bilateral_texture.png")

blur = cv2.blur(img_filter, (5, 5))
# sigmaX = 0 lets OpenCV derive the standard deviation from the kernel size.
gaussian_blur = cv2.GaussianBlur(img_filter, (5, 5), 0)
# Median filtering is the standard fix for salt-and-pepper noise.
median_blur = cv2.medianBlur(img_median, 5)
# Bilateral filtering smooths flat regions while keeping edges sharp.
bilateral = cv2.bilateralFilter(img_bilateral, 9, 152, 51)

cv2.imshow("Original", img_filter)
cv2.imshow("Blur", blur)
cv2.imshow("Gaussian blur", gaussian_blur)
cv2.imshow("Median original", img_median)
cv2.imshow("Median blur", median_blur)
cv2.imshow("Bilateral original", img_bilateral)
cv2.imshow("Bilateral filter", bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
