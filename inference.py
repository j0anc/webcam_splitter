import os
from functools import cache

import cv2
import numpy as np
from ultralytics import YOLO

from typing import Optional, List, Any


@cache
def load_model(model_path: str):
    model = YOLO(model_path)
    return model


def get_boxes(pred) -> np.ndarray:
    boxes = pred[0].boxes.data
    boxes_arr = boxes.detach().numpy().copy()
    return boxes_arr


def crop_boxes_to_image(
    img: np.ndarray,
    boxes: np.ndarray,
    mode: str,
    output_dir: str,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    for webcam_idx, box in enumerate(boxes):
        x_min, y_min, x_max, y_max, conf, _ = box

        if conf < 0.7:
            continue

        if mode == "simple":
            output_file_path = os.path.join(
                output_dir, f"webcam_{str(webcam_idx)}", f"{frame_idx}.jpg"
            )

        elif mode == "normal":
            output_file_path = os.path.join(output_dir, f"{frame_idx}_{webcam_idx}.jpg")

        elif mode == "image":
            output_file_path = os.path.join(output_dir, f"{webcam_idx}.jpg")

        else:
            print("invalid mode")

        cropped_area = img[int(y_min) : int(y_max), int(x_min) : int(x_max)]
        cv2.imwrite(output_file_path, cropped_area)

    return


def crop_boxes_to_video(
    img: np.ndarray, boxes: np.ndarray, video_writers: List[Any]
) -> None:
    for webcam_idx, box in enumerate(boxes):
        x_min, y_min, x_max, y_max, conf, _ = box

        if conf < 0.7:
            continue

        cropped_area = img[int(y_min) : int(y_max), int(x_min) : int(x_max)]
        video_writers[webcam_idx].write(cropped_area)

    return
