import cv2

# The cascades are shared with the mask detection experiments, so they live at the repository root.
face_cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
    raise IOError("Could not load the Haar cascades. Run the scripts from inside 09-face-detection/.")


def detect(frame, scale_factor=1.3, thickness=2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale_factor, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness)
        # Eyes are only searched for inside each face, which is faster and
        # avoids false positives in the background.
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return frame
