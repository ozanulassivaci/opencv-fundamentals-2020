import cv2

cap = cv2.VideoCapture(0)
file_name = "media/webcam_recording.avi"
codec = cv2.VideoWriter_fourcc(*"WMV2")
frame_rate = 30
resolution = (640, 480)

video_writer = cv2.VideoWriter(file_name, codec, frame_rate, resolution)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Mirror the preview so moving left on camera moves left on screen.
    frame = cv2.flip(frame, 1)
    # VideoWriter silently drops frames whose size differs from the declared resolution.
    frame = cv2.resize(frame, resolution)
    video_writer.write(frame)
    cv2.imshow("Webcam live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_writer.release()
cap.release()
cv2.destroyAllWindows()
