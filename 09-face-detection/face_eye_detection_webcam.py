"""Press "q" to close the window."""

import cv2

from face_eye_detector import detect

# If more than one camera is connected, change 0 to the index of the camera to use.
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    canvas = detect(frame)
    cv2.imshow("Video", canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
