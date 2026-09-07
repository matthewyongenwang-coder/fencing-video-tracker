# Vendored third-party code

## mp_pose.py
Copied unchanged from the OpenCV Model Zoo:
https://github.com/opencv/opencv_zoo/blob/main/models/pose_estimation_mediapipe/mp_pose.py

Licensed Apache 2.0 (see LICENSE-mediapipe-pose.txt). The underlying model is
Google's MediaPipe BlazePose, also Apache 2.0.

It is vendored rather than reimplemented because the preprocessing (crop,
pad-to-square, rotate to the hip->shoulder axis, then invert all of that on the
way back out) is fiddly and easy to get subtly wrong. `pose.py` wraps it.

The zoo's matching person DETECTOR (mp_persondet.py) is deliberately NOT used:
its postprocessing assumes OpenCV 4.x output shapes and crashes on OpenCV 5.
We don't need it anyway -- we already know where each fencer is, so the tracked
box is used as the region of interest instead.
