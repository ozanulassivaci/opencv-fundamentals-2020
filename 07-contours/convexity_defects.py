import cv2

img = cv2.imread("media/star.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
# convexityDefects needs hull point indices, not coordinates.
hull = cv2.convexHull(cnt, returnPoints=False)

defects = cv2.convexityDefects(cnt, hull)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]

    start = tuple(int(v) for v in cnt[s][0])
    end = tuple(int(v) for v in cnt[e][0])
    far = tuple(int(v) for v in cnt[f][0])

    cv2.line(img, start, end, (0, 255, 0), 2)
    cv2.circle(img, far, 5, (0, 255, 0), -1)

cv2.imshow("Convexity defects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
