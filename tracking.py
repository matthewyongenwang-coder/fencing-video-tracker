"""
tracking.py

Wraps OpenCV's CSRT tracker so each fencer gets their own, completely
independent tracker instance.

WHY CSRT
OpenCV ships several single-object trackers (KCF, MOSSE, CSRT, ...).
CSRT is the slowest of the common ones but the best at holding a box
through motion blur, partial occlusion and some change in size, and all
three happen in this clip during the lunge. The whole clip is 4 seconds,
so CSRT's extra cost is about 8 seconds of processing. That's a fine
trade. Measured on this Mac: ~14 frames/sec with both trackers running.

WHY NOT A NEURAL NETWORK (YOLO, pose estimation, etc.)
It would probably work better, and it is the obvious next step for the
phone app. But for a first version it would mean a much bigger install,
a much slower run, and a model that still can't tell your two fencers
apart from the background ones, because everybody is wearing the same
white kit. The hard part of this problem is IDENTITY, and manual
selection solves identity for free. Start there, and add a detector
later once tracking is trustworthy.

WHY TWO SEPARATE TRACKERS, NOT ONE "MULTI-TRACKER"
Each fencer gets their own CSRT instance, seeded once from the box you
draw. There is no shared appearance model and no re-identification step
between them, so there is no mechanism by which the two trackers could
trade targets with each other. Each one only ever looks for "the thing
that looked like this last frame" within its own search region.

THE LIMITATION, STATED PLAINLY
"They cannot swap with each other" is NOT the same as "they can never go
wrong". A CSRT tracker can still drift onto a different person who looks
similar and is physically close, and in this clip everyone is in
identical white kit. If Fencer A walks in front of a background fencer,
A's tracker may come out the other side following the wrong person, and
it will report success the whole time, because from its point of view it
never lost anything. Verified on this clip: that did not happen, checked
by eye on the exported frames. It is not guaranteed on your next clip.
That is exactly why the live display and the re-select key exist.
"""

import os
import cv2

VIT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "models", "vittrack.onnx")


def create_tracker(kind="csrt"):
    """
    Create one tracker instance.

    "csrt": OpenCV's CSRT. No extra files needed. Sticky, it holds on
               through blur and partial occlusion. Its weakness is that
               when it does go wrong it goes wrong SILENTLY, reporting
               success while following the wrong person.

    "vit": OpenCV's ViT tracker, a small pretrained neural network
               (a 700KB ONNX file in models/). This is using an existing
               released model, not training anything. Measured on your
               clips it is LESS sticky than CSRT and gives up sooner, but
               but it fails HONESTLY: it reports the loss and its
               getTrackingScore() collapses. Sometimes a tracker that
               admits defeat is worth more than one that doesn't.

    Neither is simply better. See the README for the numbers.
    """
    if kind == "vit":
        if not os.path.exists(VIT_MODEL_PATH):
            raise RuntimeError(
                f"ViT model not found at {VIT_MODEL_PATH}\n"
                "Download it once with:\n"
                "  mkdir -p models && curl -L -o models/vittrack.onnx \\\n"
                "    https://github.com/opencv/opencv_zoo/raw/main/models/"
                "object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx\n"
                "Or just use the default --tracker csrt, which needs no download.")
        params = cv2.TrackerVit_Params()
        params.net = VIT_MODEL_PATH
        return cv2.TrackerVit_create(params)

    if kind != "csrt":
        raise ValueError(f"Unknown tracker '{kind}'. Use 'csrt' or 'vit'.")

    # OpenCV has moved this API around between versions, so try the
    # places it has lived. Verified on OpenCV 5.0.0 and on the 4.x line.
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT") and hasattr(cv2.TrackerCSRT, "create"):
        return cv2.TrackerCSRT.create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError(
        "No CSRT tracker in this OpenCV build. You need "
        "opencv-contrib-python, not plain opencv-python.")


class Fencer:
    """
    One tracked fencer: a fixed label, a fixed colour, its own tracker,
    and its latest box + status.
    """

    def __init__(self, label: str, color_bgr: tuple, first_frame, init_box,
                 kind: str = "csrt"):
        self.label = label
        self.color = color_bgr
        self.kind = kind
        self.box = tuple(int(v) for v in init_box)
        self.tracker = create_tracker(kind)
        self.tracker.init(first_frame, self.box)
        self.tracked = True
        self.frames_lost = 0
        self.reinit_count = 0
        # How far the box centre moved since the previous frame, in
        # pixels. This is a MEASURED distance, not a confidence score --
        # CSRT does not give a confidence, so this code does not invent
        # one. A sudden large jump is worth your attention because it can
        # mean the box hopped to a different person, but it can equally
        # mean a fast lunge. It is a hint for you, not a verdict.
        self.last_jump_px = 0.0
        # A real confidence, ONLY when the tracker actually provides one.
        # CSRT does not expose any confidence, so for CSRT this stays
        # None rather than being filled with a made-up number.
        self.score = None

    def _read_score(self):
        getter = getattr(self.tracker, "getTrackingScore", None)
        if getter is None:
            return None
        try:
            return float(getter())
        except cv2.error:
            return None

    def update(self, frame):
        previous_center = self.center()
        ok, box = self.tracker.update(frame)
        self.tracked = bool(ok)
        self.score = self._read_score()

        if ok:
            self.box = tuple(int(v) for v in box)
            self.frames_lost = 0
            cx, cy = self.center()
            self.last_jump_px = ((cx - previous_center[0]) ** 2 +
                                 (cy - previous_center[1]) ** 2) ** 0.5
        else:
            # Keep the last known box so the label still has somewhere to
            # draw, but leave self.tracked False so nothing downstream
            # mistakes a stale box for a live one.
            self.frames_lost += 1
            self.last_jump_px = 0.0

        return self.tracked, self.box

    def center(self):
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)

    def reinit(self, frame, new_box):
        """
        Start this fencer's tracker over from a fresh box. Used when you
        pause and re-select after a loss or a visible drift.

        A brand new tracker instance is created rather than re-init'ing
        the old one, so no trace of the previous appearance model
        survives to pull the box back toward the wrong target.
        """
        self.tracker = create_tracker(self.kind)
        self.box = tuple(int(v) for v in new_box)
        self.tracker.init(frame, self.box)
        self.tracked = True
        self.frames_lost = 0
        self.last_jump_px = 0.0
        self.score = None
        self.reinit_count += 1
