import cv2

# Cascades: https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_mcs_mouth.xml")

if face_cascade.empty() or mouth_cascade.empty():
    raise IOError("Could not load the Haar cascades. Run the script from inside 10-mask-detection/.")

# Adjust in the 80-105 range depending on the lighting.
bw_threshold = 85

font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
mask_on_color = (0, 255, 0)
mask_off_color = (0, 0, 255)
thickness = 2
font_scale = 1
mask_on_text = "Thank you for wearing a mask"
mask_off_text = "Please wear a mask"

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.flip(img, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, black_and_white = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("Black and white", black_and_white)

    faces = face_cascade.detectMultiScale(gray, 1.3, 10)
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.3, 6)

    if len(faces) == 0 and len(faces_bw) == 0:
        cv2.putText(img, "No face found", org, font, font_scale, mask_on_color, thickness, cv2.LINE_AA)
    elif len(faces) == 0 and len(faces_bw) == 1:
        # A white mask over the mouth hides the face from the grayscale detector,
        # but the face is still found in the black-and-white image.
        cv2.putText(img, mask_on_text, org, font, font_scale, mask_on_color, thickness, cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 7)

            # A face was found but no lips, so the mouth is covered.
            if len(mouth_rects) == 0:
                cv2.putText(img, mask_on_text, org, font, font_scale, mask_on_color, thickness, cv2.LINE_AA)
            else:
                for (mx, my, mw, mh) in mouth_rects:
                    # The mouth cascade also fires outside the face, so only lips
                    # inside the face box count as an uncovered mouth.
                    if y < my < y + h:
                        cv2.putText(img, mask_off_text, org, font, font_scale, mask_off_color, thickness,
                                    cv2.LINE_AA)
                        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 0, 255), 3)
                        break

    cv2.imshow("Mask detection", img)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
