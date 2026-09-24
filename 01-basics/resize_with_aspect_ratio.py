import cv2


def resize_with_aspect_ratio(img, width=None, height=None, inter=cv2.INTER_AREA):
    # Only one target dimension is given; the other is scaled by the same ratio
    # so the image is not stretched.
    (h, w) = img.shape[:2]

    if width is None and height is None:
        return img

    if width is None:
        r = height / float(h)
        dimension = (int(w * r), height)
    else:
        r = width / float(w)
        dimension = (width, int(h * r))

    return cv2.resize(img, dimension, interpolation=inter)


img = cv2.imread("media/portrait_pink.png")
resized = resize_with_aspect_ratio(img, height=600)

print(f"Original: {img.shape[1]}x{img.shape[0]}, resized: {resized.shape[1]}x{resized.shape[0]}")
cv2.imshow("Original", img)
cv2.imshow("Resized", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
