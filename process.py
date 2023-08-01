import os

import cv2
from ultralytics import YOLO

from inference import crop_boxes, get_boxes, load_model
from utils import clear_create_dir, get_filename


def process_video(file_path: str, mode: str, output_type: str):
    # create output directories
    filename = get_filename(file_path)
    output_dir = os.path.join(os.getcwd(), "output", filename)
    clear_create_dir(output_dir)

    # load model
    model = load_model(os.path.join(os.getcwd(), "model", "webcam_split.pt"))

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error: Failed to open the video file.")
        return

    frame_idx = 0
    boxes = []
    while frame_idx < 50:
        ret, frame = cap.read()

        if not ret:
            break

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

                for i in range(len(boxes)):
                    os.makedirs(os.path.join(output_dir, f"webcam_{str(i)}"))

            crop_boxes(frame, boxes, "simple", output_dir, frame_idx)

            if output_type == "video":
                pass

        # can only save cropped webcam areas as image
        if mode == "normal":
            pred = model.predict(frame)
            boxes = get_boxes(pred)
            crop_boxes(frame, boxes, "normal", output_dir, frame_idx)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    return


def process_image(file_path: str):
    filename = get_filename(file_path)

    img = cv2.imread(file_path)
    model = load_model(os.path.join(os.getcwd(), "model", "webcam_split.pt"))

    pred = model.predict(img)
    boxes = get_boxes(pred)

    output_dir = os.path.join(os.getcwd(), "output", filename)
    clear_create_dir(output_dir)

    crop_boxes(img, boxes, "image", output_dir)

    return
