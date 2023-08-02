import os
import shutil

import cv2
import numpy as np
from typing import Any, List


def create_output_dir(output_dir: str) -> None:
    if os.path.exists(output_dir):
        print(f"Directory '{output_dir}' already exists.")
    else:
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    return


def clear_create_dir(dir: str) -> None:
    if os.path.exists(dir):
        print(f"Existing files in {dir} will be cleared...")
        shutil.rmtree(dir)

    os.makedirs(dir)
    print(f"Directory '{dir}' created.")
    return


def get_filename(file_path: str) -> str:
    filename, _ = os.path.splitext(os.path.basename(file_path))
    return filename


def get_video_writers(
    output_dir: str, frame_rate: int, boxes: np.ndarray, video_writers: List[Any]
) -> List[Any]:
    codec = cv2.VideoWriter_fourcc(*"mp4v")

    for i, box in enumerate(boxes):
        output_video_path = os.path.join(output_dir, f"webcam_{i}.mp4")
        frame_size = (int(box[2]) - int(box[0]), int(box[3]) - int(box[1]))
        video_writer = cv2.VideoWriter(output_video_path, codec, frame_rate, frame_size)
        video_writers.append(video_writer)

    return video_writers


def create_output_dirs(dir_count: int, output_dir: str) -> None:
    for i in range(dir_count):
        os.makedirs(os.path.join(output_dir, f"webcam_{str(i)}"))


def close_video_writers(video_writers: List[Any]) -> None:
    if len(video_writers) > 0:
        for video_writer in video_writers:
            video_writer.release()
    return
