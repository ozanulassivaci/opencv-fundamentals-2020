import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread("media/polygons.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # A tolerance of 1% of the perimeter is small enough to keep real corners
    # and large enough to merge the jagged pixels along each edge.
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)

    x = int(approx.ravel()[0])
    y = int(approx.ravel()[1])

    vertices = len(approx)
    if vertices == 3:
        label = "Triangle"
    elif vertices == 4:
        label = "Rectangle"
    elif vertices == 5:
        label = "Pentagon"
    elif vertices == 6:
        label = "Hexagon"
    else:
        label = "Ellipse"

    cv2.putText(img, label, (x, y), font, 1, (0, 0, 0))

cv2.imshow("Shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
