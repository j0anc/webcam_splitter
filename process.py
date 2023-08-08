import os

import cv2
from ultralytics import YOLO

from inference import crop_boxes_to_image, crop_boxes_to_video, get_boxes, load_model
from utils import (
    clear_create_dir,
    close_video_writers,
    create_output_dirs,
    get_filename,
    get_video_writers,
)


def process_video(file_path: str, mode: str, output_type: str, fps: float) -> None:
    # create output directories
    filename = get_filename(file_path)
    output_dir = os.path.join(os.getcwd(), "output", filename)
    clear_create_dir(output_dir)

    # load model
    model = load_model(os.path.join(os.getcwd(), "model", "webcam_split.pt"))

    cap = cv2.VideoCapture(file_path)

    # get video properties
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    if fps is None:
        fps = frame_rate

    # calculate frame interval for fps adjustment
    frame_interval = frame_rate // fps

    if not cap.isOpened():
        print("Error: Failed to open the video file.")
        return

    frame_idx = 0
    boxes = []
    video_writers = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        # can choose to save cropped webcam areas to image or save it to a video
        if mode == "simple":
            # only run webcem detection on the first frame
            if frame_idx == 0:
                pred = model.predict(frame)
                boxes = get_boxes(pred)

                if len(boxes) == 0:
                    print(
                        "no webcams detected from the first frame. use normal mode maybe?"
                    )

                if output_type == "image":
                    create_output_dirs(len(boxes), output_dir)

                if output_type == "video":
                    get_video_writers(output_dir, frame_rate, boxes, video_writers)

            if output_type == "video":
                crop_boxes_to_video(frame, boxes, video_writers)

            if output_type == "image":
                crop_boxes_to_image(frame, boxes, "simple", output_dir, frame_idx)

        # can only save cropped webcam areas as image
        if mode == "normal":
            pred = model.predict(frame)
            boxes = get_boxes(pred)
            crop_boxes_to_image(frame, boxes, "normal", output_dir, frame_idx)

        frame_idx += 1

    close_video_writers(video_writers)

    cap.release()
    cv2.destroyAllWindows()

    return


def process_image(file_path: str) -> None:
    filename = get_filename(file_path)

    img = cv2.imread(file_path)
    model = load_model(os.path.join(os.getcwd(), "model", "webcam_split.pt"))

    pred = model.predict(img)
    boxes = get_boxes(pred)

    output_dir = os.path.join(os.getcwd(), "output", filename)
    clear_create_dir(output_dir)

    crop_boxes_to_image(img, boxes, "image", output_dir)

    return
