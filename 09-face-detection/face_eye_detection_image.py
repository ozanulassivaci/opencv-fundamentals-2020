import imageio.v2 as imageio

from face_eye_detector import detect

# imageio loads images as RGB, so the (255, 0, 0) face box comes out red here
# while it is blue in the OpenCV (BGR) webcam version.
image = imageio.imread("media/selfie.jpg")
image = detect(frame=image)
imageio.imwrite("media/selfie_detection.jpg", image)
