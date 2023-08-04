# webcam_splitter

This command-line tool processes online meeting recording videos with multiple participants and splits them into data for each person. It offers the flexibility to save the output as individual images for each frame or combine the frames into new cropped videos.

## Installation

The dependencies are managed with [Poetry](https://python-poetry.org/). Simply run `poetry install` to install everything.

## Usage

```
poetry run crop <file_path> [--mode <normal/simple>] [--output-type <image/video>]
```
- file_path: Path to a video or image file
- mode: Process mode for video input files (optional). Choose between "simple" or "normal". (default: "normal")<br>
  - Simple mode: <br>
  Runs object detection (detection of each participant's webcam area) on the first frame and crops all consecutive frames with the same detection result. Use this mode if your video file contains only the same setting (i.e., same participants, all participants stay at the same area of the video throughout the entire video). The prediction runs only once, making it computationally more efficient
  - Normal mode: <br>
  Runs object detection on every frame, providing more accurate cropping if the number of participants or their positions change throughout the video. Use this mode when your video has a varying number of participants or the participants change order or places in the video
- output-type: Output type for video simple mode (optional). Choose between "image" or "video" (default: "image")

    **output directory structure for each scenario**
  
    - For image input:
      ```
      ./output/
        └── image_name/
            ├── 0.jpg
            ├── 1.jpg
            ├── 2.jpg
            ├── 3.jpg
            └── ...
      ```
      
    - For video input with normal mode (file name format: {frame_index}_{webcam_index}.jpg):
      ```
      ./output/
        └── video_name/
          ├── 0_0.jpg
          ├── 0_1.jpg
          ├── 0_2.jpg
          ├── 0_3.jpg
          ├── 1_0.jpg
          ├── 1_1.jpg
          ├── 1_2.jpg
          ├── 1_3.jpg
          ├── 2_0.jpg
          └── ...
      ```
    - For video input with simple mode and image outputs:
      ```
      ./output/
        └── video_name/
            ├── webcam_0/
            │   ├── 0.jpg
            │   ├── 1.jpg
            │   └── ...
            ├── webcam_1/
            │   ├── 0.jpg
            │   ├── 1.jpg
            │   └── ...
            └── webcam_2/
                ├── 0.jpg
                ├── 1.jpg
                └── ...
      ```
    - For video input with simple mode and video outputs:
      ```
      ./output/
        └── video_name/
            ├── webcam_0.mp4
            ├── webcam_1.mp4
            ├── webcam_2.mp4
            ├── webcam_3.mp4
            └── ...
      ```




## Object Detection in this Repo
The object detection here is based on YOLOv8, trained on online meeting recording data from various platforms. While the training data is limited, it might not perform perfectly on all data types. For better results, consider fine-tuning the model on your own dataset, adjusting hyperparameters, and expanding the training data to suit your specific use case.

## Possible Purpose of Using This Tool

The Webcam Cropping Tool is designed to efficiently gather webcam data for machine learning model training. By extracting individual webcam images from online meeting recordings, it simplifies the process of collecting data for training machine learning models that require webcam input. One video file can contain multiple people's webcam image data, making the data gathering process more efficient and convenient.
