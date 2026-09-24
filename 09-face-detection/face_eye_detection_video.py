import imageio.v2 as imageio

from face_eye_detector import detect


def process_video(input_path, output_path, scale_factor, thickness):
    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()["fps"]
    writer = imageio.get_writer(output_path, fps=fps)
    for i, frame in enumerate(reader):
        writer.append_data(detect(frame, scale_factor, thickness))
        print(f"{input_path}: frame {i}")
    writer.close()


process_video("../03-video-io/media/webcam_recording.avi", "media/webcam_recording_detection.avi",
              scale_factor=1.1, thickness=5)
process_video("media/portrait_video.mp4", "media/portrait_video_detection.mp4",
              scale_factor=1.3, thickness=2)
