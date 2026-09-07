"""
pose.py

Finds body landmarks -- the "lines drawn on the body" you were thinking of --
for each fencer we are already tracking.

WHAT THIS IS
BlazePose, a small pretrained model from Google, released by OpenCV in
their Model Zoo. It returns 33 body points: eyes, shoulders, elbows,
wrists, hips, knees, ankles, feet. It runs through cv2.dnn, so it adds
NO new pip dependency -- just a 5MB model file, the same arrangement as
the ViT tracker.

WHY IT IS POINTED AT OUR BOX INSTEAD OF THE WHOLE FRAME
BlazePose expects one person, roughly centred. The zoo ships a person
detector to find them, but that detector's postprocessing assumes
OpenCV 4 output shapes and crashes on OpenCV 5. We don't need it: we
already know where each fencer is, because you selected them and the
tracker has been following them. So the tracked box becomes the region
of interest. This also means the pose is always attached to the fencer
YOU chose, instead of whichever body the detector happened to rank
first -- which matters a lot in a hall full of identical white kit.

WHAT IT GIVES US THAT A BOX DOES NOT
* ANKLES. Where the fencer's feet actually meet the piste. A box edge is
  arbitrary -- it moves when the box drifts or resizes. Feet are real.
  This is the anchor that makes "has the tracker wandered into the
  background?" answerable, because background people's feet sit much
  higher in frame than piste-level feet.
* WRISTS AND ELBOWS. The sword hand, and how extended the arm is. That
  is the closest honest handle we have on blade actions -- see the
  README about why the blade itself mostly isn't there to be seen.
* A REAL "IS ANYONE HERE?" ANSWER. Point it at empty floor and it
  returns nothing. Verified: a box on bare piste gives no pose at all,
  while a box on either fencer gives confidence 0.94-1.00.

MEASURED COST
About 27 frames/sec for one fencer on this Mac, so roughly halved for
two. On a 4 second clip that's a few extra seconds. It is off by default
(`--pose`) because you don't always need it.

WHAT IT DOES NOT DO HERE
It does NOT drive the tracker. It was tempting -- letting pose re-lock
the box fixed a drift that CSRT could not survive, moving Fencer A from
completely lost to correct. But the same change made the OTHER fencer
worse, so it is not a net win yet and is not wired in. The numbers are
in the README. For now pose MEASURES and WARNS; it does not steer.
"""

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
from mp_pose import MPPose   # noqa: E402  (vendored, see vendor/README.md)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "pose_landmarks.onnx")

# BlazePose landmark numbers we actually use.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# Lines to draw. Keeps the stick figure readable without drawing all 33.
SKELETON = [
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]


class PoseResult:
    """One fencer's landmarks on one frame, in full-frame pixels."""

    def __init__(self, landmarks, confidence):
        self.landmarks = landmarks          # (33, 2) x,y in frame pixels
        self.visibility = None              # (33,) how sure of each point
        self.confidence = float(confidence)

    @property
    def feet(self):
        """Midpoint between the ankles -- where the fencer meets the floor."""
        return tuple((self.landmarks[LEFT_ANKLE] + self.landmarks[RIGHT_ANKLE]) / 2.0)

    @property
    def hips(self):
        return tuple((self.landmarks[LEFT_HIP] + self.landmarks[RIGHT_HIP]) / 2.0)

    @property
    def shoulders(self):
        return tuple((self.landmarks[LEFT_SHOULDER] +
                      self.landmarks[RIGHT_SHOULDER]) / 2.0)

    @property
    def torso_size(self):
        """
        Shoulder-to-opposite-hip distance. A rough stand-in for how large
        the fencer appears, which is far steadier than the box size.
        """
        return float(np.linalg.norm(self.landmarks[LEFT_SHOULDER] -
                                    self.landmarks[RIGHT_HIP]))

    def sword_hand(self, prefer_right=True):
        """
        The wrist further from the body centre -- the extended arm, which
        for a fencer en garde is the sword arm.

        This is a GUESS from geometry. It does not know which hand holds
        the sabre, and a left-hander or a moment with both arms out will
        fool it. It is a starting point, not a fact.
        """
        centre = np.array(self.hips)
        left = np.linalg.norm(self.landmarks[LEFT_WRIST] - centre)
        right = np.linalg.norm(self.landmarks[RIGHT_WRIST] - centre)
        idx = LEFT_WRIST if left >= right else RIGHT_WRIST
        elbow = LEFT_ELBOW if idx == LEFT_WRIST else RIGHT_ELBOW
        return tuple(self.landmarks[idx]), tuple(self.landmarks[elbow])

    def arm_extension(self):
        """
        How far the sword hand reaches from the shoulders, measured in
        torso-widths so it doesn't change just because the fencer is
        further from the camera.

        A bigger number means a more extended arm. It does NOT mean an
        attack, a parry, or a hit -- it is arm geometry, nothing more.
        """
        wrist, _elbow = self.sword_hand()
        reach = np.linalg.norm(np.array(wrist) - np.array(self.shoulders))
        size = self.torso_size
        return float(reach / size) if size > 1.0 else 0.0


class PoseEstimator:
    """
    Runs BlazePose on the crop around a tracked box.

    Returns None when it cannot find a person, which is genuinely useful:
    a box sitting on empty floor produces None.
    """

    def __init__(self, confidence_threshold=0.4):
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Pose model not found at {MODEL_PATH}\n"
                "Download it once with:\n"
                "  mkdir -p models && curl -L -o models/pose_landmarks.onnx \\\n"
                "    https://github.com/opencv/opencv_zoo/raw/main/models/"
                "pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx\n"
                "Or run without --pose.")
        self.model = MPPose(modelPath=MODEL_PATH,
                            confThreshold=confidence_threshold)

    @staticmethod
    def _region_hint(box, hip_fraction=0.60, reach=1.15):
        """
        Build the input the vendored code expects.

        It normally comes from a person detector: two points, roughly
        mid-hip and a point out past the head, which together set both
        the square crop and the rotation. We fake it from the tracked
        box -- hips a little below box centre, "up" being straight up,
        and a square generous enough to include legs and feet even when
        you only boxed the torso.
        """
        x, y, w, h = box
        mid_hip = np.array([x + w / 2.0, y + h * hip_fraction])
        distance = max(w, h) * reach
        top = mid_hip + np.array([0.0, -distance])
        hint = np.zeros(16, np.float32)
        hint[4:6] = mid_hip
        hint[6:8] = top
        hint[8:10] = mid_hip
        hint[10:12] = top
        return hint

    def estimate(self, frame, box):
        """Returns a PoseResult, or None if no person was found."""
        if box[2] < 10 or box[3] < 10:
            return None
        # _region_hint returns a fresh array each call on purpose: the
        # vendored preprocessing subtracts the pad offset in place, so a
        # reused array would be quietly corrupted after the first frame.
        raw = self.model.infer(frame, self._region_hint(box))
        if raw is None:
            return None
        _bbox, landmarks, _world, _mask, _heatmap, confidence = raw
        result = PoseResult(landmarks[:33, :2].astype(np.float32), confidence)
        result.visibility = landmarks[:33, 3].astype(np.float32)
        return result


class FootWatch:
    """
    Warns when a fencer's feet stop behaving like feet on a piste.

    THE IDEA
    A fencer moves along the floor. Their feet trace a smooth path, and
    because of perspective that path drifts only gradually up or down the
    frame. A tracker that has wandered onto someone in the background
    lands on feet that are suddenly much HIGHER in the frame, because
    distant people's feet sit higher up.

    Measured on the clip where Fencer A drifted onto the referee:
        frames   0- 24   feet at y ~795   (correct, on the near piste)
        frames 108-216   feet at y ~500-600 (background people)
    A ~290 pixel rise that never comes back is not something a fencer can
    do on a flat floor. Meanwhile the fencer who stayed correct moved
    only ~65px over the same clip.

    TWO BUGS THIS WENT THROUGH, BOTH WORTH KNOWING ABOUT
    1. The first version compared against a fast-adapting running
       average. The average simply FOLLOWED the drift down -- by the time
       the feet had climbed 250px the reference had climbed with them and
       the gap looked tiny. It never fired. A reference you are trying to
       detect movement away from must not chase the movement.
    2. It also reset its evidence counter every frame the pose model
       found nothing. Pose only lands on about 75% of frames here (it
       fails during the blurriest moments), so the gaps wiped the count
       before it could ever reach the threshold. A missing measurement is
       not evidence that everything is fine.

    So the reference is now the median of the OLDEST third of a 3-second
    window -- old enough that the drift has not polluted it -- and gaps
    hold the evidence rather than clearing it.

    HONESTY ABOUT THIS CHECK
    It assumes the camera does not tilt much, and it is tuned against
    exactly one real drift plus three correct runs. Every threshold from
    120px to 200px separated those four cases, so it is not knife-edge,
    but four runs is not a lot. Pose confidence, for what it is worth,
    does NOT work for this at all -- it sat at 0.97+ while tracking the
    wrong people, because those people are perfectly real.
    """

    def __init__(self, window=90, rise_px=150, frames_before_warning=10):
        # 90 frames is about 3 seconds at 30fps.
        self.window = window
        self.rise_px = rise_px
        self.frames_before_warning = frames_before_warning
        self.history = deque(maxlen=window)
        self.consecutive = 0
        self.warned = False
        self.reference = None

    def update(self, pose_result):
        """Returns (feet_y or None, is_warning, is_new_warning)."""
        warning_now = self.consecutive >= self.frames_before_warning

        if pose_result is None:
            # Hold the evidence: no reading is not a reading of "fine".
            return None, warning_now, False

        feet_y = pose_result.feet[1]

        if len(self.history) < self.window:
            self.history.append(feet_y)
            return feet_y, False, False

        # Compare against the oldest third of the window, which is old
        # enough not to have been dragged along by a recent drift.
        old = list(self.history)[: self.window // 3]
        self.reference = float(np.median(old))

        if self.reference - feet_y > self.rise_px:
            self.consecutive += 1
        else:
            self.consecutive = 0
            self.warned = False

        self.history.append(feet_y)

        warning = self.consecutive >= self.frames_before_warning
        new_warning = warning and not self.warned
        if new_warning:
            self.warned = True
        return feet_y, warning, new_warning

    def reset(self):
        self.history.clear()
        self.consecutive = 0
        self.warned = False
        self.reference = None
