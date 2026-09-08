# Third-party code and models

This project is MIT licensed (see [LICENSE](LICENSE)). It also includes
third-party code, and downloads third-party models, which carry their own
terms.

## Vendored code

**`vendor/mp_pose.py`** is copied unchanged from the
[OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/blob/main/models/pose_estimation_mediapipe/mp_pose.py),
and is licensed Apache 2.0. The full licence text is in
[`vendor/LICENSE-mediapipe-pose.txt`](vendor/LICENSE-mediapipe-pose.txt).

I vendored it rather than rewriting it because its preprocessing (crop,
pad to square, rotate to the hip-to-shoulder axis, then undo all of that on
the way back out) is fiddly and easy to get subtly wrong.

## Models

`get_models.py` downloads three pretrained models at setup time. None of
them are redistributed in this repository and I did not train any of them.
All three are released models used as published.

| Model | Source | Used for |
|---|---|---|
| YOLOX-S | [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox) | Person detection (`--detect`) |
| MediaPipe BlazePose | [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/pose_estimation_mediapipe) | Body landmarks (`--pose`) |
| ViT tracker | [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack) | Alternative tracker (`--tracker vit`) |

Each carries the licence stated in its Model Zoo directory. Check those
terms before redistributing the model files or using them commercially.

## Test footage

The clips referenced in `benchmark.py` are my own competition recordings
and are not included in this repository. Point `FENCING_VIDEOS` at your own
footage, or edit `CASES`.
