"""
motion.py

Turns tracked positions into simple Forward / Backward / Still labels.

WHAT THESE LABELS MEAN
"Forward" = this fencer's box moved toward the other fencer along the
piste, faster than a small threshold. That is all. It is a statement
about pixels moving on screen.

WHAT THESE LABELS DO NOT MEAN
They do not identify an attack, a parry, a riposte, a hit, right of way,
or who won the point. A fencer moving forward might be attacking,
closing distance, or falling over. Blade actions are what separate those,
and this prototype does not look at the blade at all. Do not read a
refereeing decision out of this column.

TWO IMPROVEMENTS OVER THE FIRST VERSION
1. The piste direction is now MEASURED, not assumed. The old code
   assumed the piste ran along the image's x-axis and hardcoded
   "left fencer forward = +x". Instead we take the line from Fencer A's
   first box to Fencer B's first box and call that the piste. If you
   film from a corner, or in portrait, or the piste sits at an angle in
   frame, this still works with no code change.
2. Speed is measured over a short time WINDOW instead of a single frame
   gap. One-frame differences on 1080p footage are mostly noise; a
   ~0.1 second window smooths that out without blurring real footwork.
"""

from collections import deque
import numpy as np

# A fencer counts as "Still" below this speed, expressed as a fraction of
# the DISTANCE BETWEEN THE TWO FENCERS when you selected them.
#
# WHY THAT REFERENCE AND NOT THE BOX SIZE:
# This used to be a fraction of the fencer's own box height. That broke
# as soon as the advice changed to "box the torso, not the whole body" --
# a torso box is about half the height of a full-body box, so the
# threshold silently halved and the labels started flickering between
# Still and Forward on noise.
#
# The gap between the two fencers at the start does not depend on how you
# drew the boxes. It is roughly fencing distance, so it is a stable
# stand-in for the scale of the scene. Measured on the two clips here it
# came out at 1258px and 1233px -- consistent, as you would hope.
#
# 0.05 x ~1250px is about 63 px/sec, which reproduces the clean labelling
# the full-body version happened to get. It is HAND-TUNED, not calibrated
# against a ground truth of when advances really happened.
STILL_FRACTION_OF_SEPARATION = 0.05

# How far back to look when measuring speed, in seconds.
SPEED_WINDOW_SEC = 0.10

# A new label has to survive this many frames in a row before it
# replaces the current one.
#
# WHY: at 30fps, one frame is 33 milliseconds. A fencer cannot reverse
# direction in 33ms -- that is not footwork, it is the tracking box
# wobbling during a blurry lunge. Without this, the labels flickered
# Forward/Backward/Still on single frames right in the middle of the
# most interesting part of the clip. 2 frames (~67ms) is still far
# quicker than any real change of direction, so genuine footwork is not
# smoothed away.
MIN_HOLD_FRAMES = 2


class PisteAxis:
    """
    The direction of the piste, worked out from where you put the two
    boxes on the first frame.
    """

    def __init__(self, center_a, center_b):
        vec = np.array(center_b, dtype=float) - np.array(center_a, dtype=float)
        length = float(np.linalg.norm(vec))
        if length < 1.0:
            raise ValueError(
                "Fencer A and Fencer B boxes are on top of each other, so the "
                "piste direction can't be worked out. Re-select them.")
        self.unit = vec / length
        self.separation_px = length

    def project(self, point) -> float:
        """
        Squash a 2-D point down to a single number: how far along the
        piste it sits. Increasing = toward Fencer B's starting side.
        """
        return float(point[0] * self.unit[0] + point[1] * self.unit[1])

    def describe(self) -> str:
        dx, dy = self.unit
        return (f"piste axis = ({dx:+.3f}, {dy:+.3f}) in image coords, "
                f"fencers {self.separation_px:.0f}px apart at selection")


class MovementLabeler:
    """
    One per fencer. Feed it stabilised positions, get back a label.

    'sign' is +1 for the fencer whose "toward the opponent" direction is
    the positive piste direction (Fencer A), and -1 for the other one.
    """

    def __init__(self, axis: PisteAxis, sign: int):
        self.axis = axis
        self.sign = sign
        self.reset()

    def reset(self):
        """
        Forget everything. Called at the start, and again whenever you
        re-select this fencer, so nothing from before the re-selection
        leaks into the numbers after it.
        """
        self.history = deque()          # (time_sec, position_along_piste)
        self.current_label = "Still"    # what we're reporting right now
        self.pending_label = None       # a candidate waiting to be confirmed
        self.pending_frames = 0

    def _debounce(self, raw_label):
        """Only accept a new label once it has held for MIN_HOLD_FRAMES."""
        if raw_label == self.current_label:
            self.pending_label, self.pending_frames = None, 0
            return self.current_label

        if raw_label == self.pending_label:
            self.pending_frames += 1
        else:
            self.pending_label, self.pending_frames = raw_label, 1

        if self.pending_frames >= MIN_HOLD_FRAMES:
            self.current_label = raw_label
            self.pending_label, self.pending_frames = None, 0

        return self.current_label

    def update(self, time_sec, stabilised_center, tracked):
        """
        Returns (label, signed_speed_px_per_sec).

        Speed is positive when moving toward the opponent, negative when
        moving away. It is in pixels per second -- an on-screen quantity,
        not a real-world one. Converting it to metres per second would
        need camera calibration and a known reference length on the
        piste, which this prototype does not do.
        """
        if not tracked:
            # Throw away the history. Otherwise, after a gap of lost
            # frames, we'd compare a fresh position against a stale one
            # and report a huge fake speed. (The first version had this
            # bug: it divided a long-gap distance by a one-frame time.)
            # reset() leaves current_label as "Still", deliberately. We
            # return "Lost" for this frame, but the moment tracking comes
            # back the debounce must not spend two more frames still
            # insisting the fencer is lost.
            self.reset()
            return "Lost", 0.0

        position = self.axis.project(stabilised_center)
        self.history.append((time_sec, position))

        # Drop samples older than the window, but always keep at least
        # two so we can still measure something.
        while len(self.history) > 2 and \
                (time_sec - self.history[0][0]) > SPEED_WINDOW_SEC:
            self.history.popleft()

        if len(self.history) < 2:
            return self._debounce("Still"), 0.0

        t_old, pos_old = self.history[0]
        dt = time_sec - t_old
        if dt <= 0:
            return self._debounce("Still"), 0.0

        speed = (position - pos_old) / dt          # px/sec along the piste
        speed_toward_opponent = speed * self.sign

        threshold = STILL_FRACTION_OF_SEPARATION * self.axis.separation_px
        if abs(speed_toward_opponent) < threshold:
            raw = "Still"
        else:
            raw = "Forward" if speed_toward_opponent > 0 else "Backward"

        # The speed returned is always the raw measurement. Only the
        # LABEL is debounced, so the CSV keeps the real numbers and you
        # can re-derive labels differently later if you want to.
        return self._debounce(raw), speed_toward_opponent
