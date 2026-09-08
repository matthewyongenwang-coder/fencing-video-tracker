"""
camera.py

Estimates how the CAMERA moved between two frames, so that camera
movement doesn't get counted as fencer movement.

THE PROBLEM THIS SOLVES
If the camera drifts 10 pixels right, everything in frame (both
fencers, the background, the floor) appears to move 10 pixels left. A
tracker reports that as the fencer moving. Without correcting for it,
"Fencer A stepped back" and "the camera panned" look identical.

WHY THIS IS NOW A FULL TRANSFORM, NOT JUST SLIDING
The first version only measured the camera sliding left/right/up/down.
That was fine for the first clip, which turned out to be almost perfectly
still (0.05 degrees of rotation, 7 pixels of drift over 4 seconds --
iPhone stabilisation had already done the work).

Then a second clip was measured, one where you panned to follow a
fleche:

    clip 1 (still) : rotation 0.05 deg | zoom 0.9994x | drift   7 px
    clip 4 (pan)   : rotation 1.01 deg | zoom 1.0924x | drift 164 px

A 9.2% zoom is not a rounding error. Sliding-only correction ignores it
completely, and near the frame edges a 9% zoom moves things by dozens of
pixels. So this now estimates a full similarity transform: slide,
rotate, and zoom together.

HOW IT WORKS (plain version)
1. Pick a few hundred easy-to-recognise corner points on the floor and
   walls, things that genuinely don't move.
2. Ignore anything inside the fencers' boxes, so the fencers' own
   movement can't pollute the estimate.
3. Find where those points went in the next frame (sparse optical flow,
   Lucas-Kanade).
4. Fit one slide+rotate+zoom transform that best explains all of them,
   using RANSAC so a handful of points that latched onto a moving person
   get thrown out as outliers rather than bending the answer.

We then keep a running transform back to the FIRST frame, so any point
can be converted into "where would this be if the camera had never
moved". That is what the fencer positions get run through.

LIMITATIONS, STILL REAL
* No perspective or parallax. The gym floor stretches away from the
  camera, so when a camera moves sideways, near things shift more than
  far things. One transform cannot be right for both depths at once, so
  it is a compromise across the frame.
* Long clips drift. Each frame's small error accumulates into the
  running transform. Over a few seconds this is minor; over a few
  minutes it is not.
* If too few points survive, this reports reliable=False and main.py
  falls back to the last good transform rather than trusting a bad one.
  It says so in the CSV instead of hiding it.
* This corrects for camera movement. It does NOT let you measure
  real-world distance or speed. Everything stays in pixels.
"""

import cv2
import numpy as np

FEATURE_PARAMS = dict(maxCorners=500, qualityLevel=0.01,
                      minDistance=15, blockSize=7)

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Below this many surviving points we don't trust the estimate.
MIN_POINTS_FOR_TRUST = 25

# Grow the fencer boxes by this fraction before masking them out, so a
# blade or trailing foot just outside the box still gets excluded.
BOX_MARGIN = 0.25


def _to_3x3(matrix_2x3):
    return np.vstack([matrix_2x3, [0.0, 0.0, 1.0]])


class CameraMotion:
    """
    Tracks camera movement and converts screen positions into positions
    as they would be if the camera had never moved.

        cam = CameraMotion(first_frame)
        info = cam.update(next_frame, fencer_boxes)
        x, y = cam.stabilise((box_centre_x, box_centre_y))
    """

    def __init__(self, first_frame):
        self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.prev_pts = None
        # Maps CURRENT frame coordinates back into FIRST frame
        # coordinates. Starts as "do nothing".
        self.to_reference = np.eye(3)

    # -- internals ------------------------------------------------------

    def _background_mask(self, shape, boxes):
        """White where we may look for features, black over the fencers."""
        mask = np.full(shape[:2], 255, dtype=np.uint8)
        for (x, y, w, h) in boxes:
            mx, my = int(w * BOX_MARGIN), int(h * BOX_MARGIN)
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1, y1 = min(shape[1], x + w + mx), min(shape[0], y + h + my)
            mask[y0:y1, x0:x1] = 0
        return mask

    def _detect(self, gray, boxes):
        mask = self._background_mask(gray.shape, boxes)
        return cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)

    def _failed(self, gray, n_points):
        """Keep the last good transform rather than inventing a new one."""
        self.prev_gray = gray
        self.prev_pts = None
        return {"dx": 0.0, "dy": 0.0, "rotation_deg": 0.0, "scale": 1.0,
                "reliable": False, "points": n_points}

    # -- public ---------------------------------------------------------

    def update(self, frame, fencer_boxes):
        """
        Measure this frame's camera movement and fold it into the running
        transform.

        The reported dx / dy are the CAMERA's motion, which is the
        opposite of how the background appears to slide. If the floor
        slides 12px right in the image, the camera panned 12px left, and
        this reports -12.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_pts is None or len(self.prev_pts) < MIN_POINTS_FOR_TRUST * 2:
            self.prev_pts = self._detect(self.prev_gray, fencer_boxes)

        if self.prev_pts is None or len(self.prev_pts) < MIN_POINTS_FOR_TRUST:
            return self._failed(gray, 0)

        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **LK_PARAMS)
        if next_pts is None:
            return self._failed(gray, 0)

        keep = status.ravel() == 1
        old_good = self.prev_pts.reshape(-1, 2)[keep]
        new_good = next_pts.reshape(-1, 2)[keep]
        if len(old_good) < MIN_POINTS_FOR_TRUST:
            return self._failed(gray, len(old_good))

        # One slide+rotate+zoom that explains the background's motion.
        # RANSAC discards points that disagree (e.g. ones that stuck to a
        # person walking through the shot).
        matrix, _inliers = cv2.estimateAffinePartial2D(
            old_good, new_good, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if matrix is None:
            return self._failed(gray, len(old_good))

        forward = _to_3x3(matrix)              # previous frame -> this frame
        try:
            backward = np.linalg.inv(forward)  # this frame -> previous frame
        except np.linalg.LinAlgError:
            return self._failed(gray, len(old_good))

        # Chain it on, so to_reference always maps THIS frame back to the
        # very first frame.
        self.to_reference = self.to_reference @ backward

        a, b = matrix[0, 0], matrix[1, 0]
        scale = float(np.hypot(a, b))
        rotation_deg = float(np.degrees(np.arctan2(b, a)))

        self.prev_gray = gray
        self.prev_pts = new_good.reshape(-1, 1, 2)

        return {
            # Negated: the camera moves opposite to the background.
            "dx": -float(matrix[0, 2]),
            "dy": -float(matrix[1, 2]),
            "rotation_deg": rotation_deg,
            "scale": scale,
            "reliable": True,
            "points": int(len(old_good)),
        }

    def to_image(self, reference_point):
        """
        The reverse of stabilise(): take a position in the steady
        reference frame and work out where it appears on THIS frame.

        Needed for predicting where a fencer should be. Motion has to be
        predicted in the steady frame (otherwise a camera pan looks like
        the fencer sprinting), but the answer has to come back to image
        coordinates to be compared against detections.
        """
        inverse = np.linalg.inv(self.to_reference)
        vec = np.array([reference_point[0], reference_point[1], 1.0])
        out = inverse @ vec
        return (float(out[0]), float(out[1]))

    def stabilise(self, point):
        """
        Convert a position in THIS frame into where it would be if the
        camera had never moved since frame 0.

        A fencer standing perfectly still gives the same stabilised
        position on every frame, no matter how much the camera pans,
        rotates or zooms.
        """
        vec = np.array([point[0], point[1], 1.0])
        out = self.to_reference @ vec
        return (float(out[0]), float(out[1]))
