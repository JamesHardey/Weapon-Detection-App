import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

def load_yolo(model_path):
    model = YOLO(model_path)
    return model

def detect_objects(frame):
    model = load_yolo("weights/best.pt")
    res = model.predict(frame, conf=0.5)
    res_plotted = res[0].plot()
    return res_plotted



def main():
    st.title("Weapon Detection System")

    detection_mode = st.radio("Select detection mode:", ["Image", "Video"])

    if detection_mode == "Image":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg" if uploaded_file.type == "image/jpeg" else ".png")
            temp_file.write(uploaded_file.read())
            temp_file.close()

            # Read the uploaded image using OpenCV
            image = cv2.imread(temp_file.name)

            # Perform object detection on the image
            detected_image = detect_objects(image)

            st.image(detected_image, channels="BGR", caption="Object Detection Result")

    elif detection_mode == "Video":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            temp_file.close()

            # Open the temporary file using cv2.VideoCapture
            video = cv2.VideoCapture(temp_file.name)

            fps = int(video.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            output_file = "output.avi"
            out = cv2.VideoWriter(output_file, codec, fps, (int(video.get(3)), int(video.get(4))))

            # Initialize progress bar
            progress_bar = st.progress(0)

            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_index in range(frame_count):
                ret, frame = video.read()
                if not ret:
                    break

                frame = detect_objects(frame)
                out.write(frame)

                # Update progress bar
                progress = (frame_index + 1) / frame_count
                progress_bar.progress(progress)

            video.release()
            out.release()
            st.success("Object detection complete!")

            # Display the processed output video
            st.video(output_file)


if __name__ == "__main__":
    main()
