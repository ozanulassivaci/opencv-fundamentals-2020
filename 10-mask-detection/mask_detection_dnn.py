# Adapted from https://github.com/chandrikadeb7/Face-Mask-Detection
import os
import sys

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

PROTOTXT_PATH = "face_detector/deploy.prototxt"
WEIGHTS_PATH = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
MASK_MODEL_PATH = "mask_detector.model"


def detect_and_predict_mask(frame, face_net, mask_net):
    # Build a blob from the frame; the mean values are the ones the SSD face detector was trained with.
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Discard weak detections
        if confidence > 0.5:
            # Box coordinates are returned relative to the frame size
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Keep the box inside the frame
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            # MobileNetV2 expects 224x224 RGB input
            face = frame[start_y:end_y, start_x:end_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((start_x, start_y, end_x, end_y))

    # Predict all faces in one batch instead of one call per face
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    return locs, preds


if not os.path.exists(MASK_MODEL_PATH):
    sys.exit(f"{MASK_MODEL_PATH} not found. This script needs a trained mask classifier; see the README.")

face_net = cv2.dnn.readNet(PROTOTXT_PATH, WEIGHTS_PATH)
mask_net = load_model(MASK_MODEL_PATH)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (locs, preds) = detect_and_predict_mask(frame, face_net, mask_net)

    for (box, pred) in zip(locs, preds):
        (start_x, start_y, end_x, end_y) = box
        (mask, without_mask) = pred

        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = f"{label}: {max(mask, without_mask) * 100:.2f}%"

        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
