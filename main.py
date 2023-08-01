import argparse
import os

from utils import create_output_dir
from process import process_video, process_image


def main():
    parser = argparse.ArgumentParser(description="webcam cropping tool")
    parser.add_argument("file_path", type=str, help="Path to a video or image file")
    parser.add_argument(
        "--mode",
        choices=["normal", "simple"],
        default="normal",
        help="Mode for video files: normal or simple (default: normal)",
    )
    parser.add_argument(
        "--output_type",
        choices=["image", "video"],
        default="image",
        help="Output type for simple mode: image or video (default: image)",
    )

    args = parser.parse_args()

    file_path = args.file_path
    mode = args.mode
    output_type = args.output_type

    create_output_dir(os.path.join(os.getcwd(), "output"))

    if file_path.lower().endswith((".mp4", ".avi", ".mov")):
        process_video(file_path, mode, output_type="image")

    elif file_path.lower().endswith((".jpg", ".jpeg", ".png")):
        process_image(file_path)

    else:
        print(
            "Error: Unsupported file format. Supported formats: mp4, avi, mov, jpg, jpeg, png"
        )
        return


if __name__ == "__main__":
    main()
