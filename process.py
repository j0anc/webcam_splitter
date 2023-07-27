import os
from functools import cache

import cv2
import numpy as np
from ultralytics import YOLO

from utils import clear_create_dir


@cache
def load_model(model_path: str):
    model = YOLO(model_path)
    return model


def detect_webcam():
    return


def crop_webcam(
    img: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int
) -> np.ndarray:
    return img[ymin:ymax, xmin:xmax]


def process_video(file_path: str, output_type: str):
    return


def process_image(file_path: str):
    filename, _ = os.path.splitext(os.path.basename(file_path))

    img = cv2.imread(file_path)
    model = load_model(os.path.join(os.getcwd(), "model", "webcam_split.pt"))

    pred = model.predict(file_path)
    boxes = pred[0].boxes.data

    boxes_arr = boxes.detach().numpy().copy()

    output_dir = os.path.join(os.getcwd(), "output", filename)
    clear_create_dir(output_dir)

    for i, box in enumerate(boxes_arr):
        x_min, y_min, x_max, y_max, conf, _ = box
        if conf > 0.7:
            cropped_area = img[int(y_min) : int(y_max), int(x_min) : int(x_max)]
            cv2.imwrite(
                os.path.join(output_dir, f"{filename}_{i}.jpg"),
                cropped_area,
            )

    return
