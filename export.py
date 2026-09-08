"""
export.py

Writing the two outputs: an annotated video and a CSV of the numbers.

TWO REAL BUGS FIXED HERE, BOTH FOUND BY MEASURING RATHER THAN ASSUMING:

1. OUTPUT FRAME RATE.
   The first version passed cv2.CAP_PROP_FPS straight into the writer.
   On this clip that property reports 28.20 fps, but the frames are
   really 29.97 fps apart. The exported video came out 4.256 seconds
   long instead of 4.003, which is 6.3% slow motion. main.py now passes in the
   fps measured from the real timestamps instead.

2. OUTPUT CODEC.
   The first version used the "mp4v" fourcc and its README admitted it
   had never been checked in QuickTime. Both were tested here on this
   Mac using AVFoundation, which is the framework QuickTime Player
   actually uses to decode:
       mp4v -> plays, but decoded in SOFTWARE (it is MPEG-4 Part 2, a
               legacy codec from the DivX era)
       avc1 -> plays, decoded in HARDWARE (this is H.264)
   Both work, so this is not a "the old one was broken" fix. But avc1
   is the better default: it's hardware-accelerated, it's what iPhones
   and every editor expect, and it matters when you move to the phone
   app. We try avc1 first and fall back to mp4v if a future OpenCV
   build can't encode H.264.
"""

import csv
import os
import cv2

# Tried in order. The first one whose writer actually opens wins.
CODEC_PREFERENCE = ["avc1", "mp4v"]


def open_writer(path: str, fps: float, frame_size: tuple):
    """
    Open a VideoWriter, trying codecs in order of preference.

    Returns (writer, fourcc_name). Raises if none of them work.

    cv2.VideoWriter never raises on a bad codec. It quietly hands back
    an object whose isOpened() is False and then silently swallows every
    frame you write, leaving a 0-byte file. So isOpened() is checked
    here rather than trusted.
    """
    last_error = None
    for fourcc_name in CODEC_PREFERENCE:
        if os.path.exists(path):
            os.remove(path)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        if writer.isOpened():
            return writer, fourcc_name
        writer.release()
        last_error = fourcc_name

    raise IOError(
        f"Could not open a video writer for {path}. "
        f"Last codec tried: {last_error}. "
        "Your OpenCV build may be missing video encoding support. "
        "reinstall with: pip install --force-reinstall opencv-contrib-python")


def verify_output(path: str, expected_frames: int) -> dict:
    """
    Re-open the file we just wrote and actually decode it.

    "The file exists and is not 0 bytes" is not proof that anything can
    play it. This decodes every frame back and counts them, which is a
    genuine check that the file is well-formed video.

    (QuickTime playability specifically was confirmed separately using
    macOS's own AVFoundation framework. See the README.)
    """
    if not os.path.exists(path):
        return {"ok": False, "reason": "file was not created"}

    size_bytes = os.path.getsize(path)
    if size_bytes == 0:
        return {"ok": False, "reason": "file is 0 bytes"}

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False, "reason": "written file cannot be re-opened",
                "size_bytes": size_bytes}

    decoded = 0
    while cap.grab():
        decoded += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return {
        "ok": decoded == expected_frames,
        "size_bytes": size_bytes,
        "frames_decoded": decoded,
        "expected_frames": expected_frames,
        "fps_in_file": fps,
        "duration_sec": decoded / fps if fps > 0 else 0.0,
        "reason": "" if decoded == expected_frames
                  else f"decoded {decoded} frames, expected {expected_frames}",
    }


CSV_COLUMNS = [
    "frame_index",
    "timestamp_ms",
    # Camera shake estimate for this frame (see camera.py).
    "camera_dx_px",
    "camera_dy_px",
    "camera_rotation_deg",
    "camera_scale",
    "camera_estimate_reliable",
    "camera_points_used",
    # How much the two fencers' boxes overlap (0 = apart, 1 = identical).
    # A sustained high value means at least one tracker is on the wrong
    # person, because two fencers cannot be in the same place.
    "box_overlap_iou",
    "overlap_warning",
]
for tag in ("a", "b"):
    CSV_COLUMNS += [
        f"fencer_{tag}_tracked",
        f"fencer_{tag}_x", f"fencer_{tag}_y",
        f"fencer_{tag}_w", f"fencer_{tag}_h",
        # Box centre exactly as it appears on screen, camera shake included.
        f"fencer_{tag}_center_x", f"fencer_{tag}_center_y",
        # Same centre after camera shake has been subtracted out. Use
        # THIS one for movement analysis and for the robot project later.
        f"fencer_{tag}_stable_x", f"fencer_{tag}_stable_y",
        # Speed along the piste, positive = toward the opponent.
        # Pixels per second, not metres per second. There is no calibration.
        f"fencer_{tag}_speed_px_s",
        f"fencer_{tag}_movement",
        # Only filled in when the tracker actually reports a confidence.
        # CSRT does not, so this column is blank for CSRT rather than
        # containing an invented number.
        f"fencer_{tag}_score",
        # How far the box jumped this frame. A measured distance, not a
        # confidence score. Large values are worth eyeballing.
        f"fencer_{tag}_box_jump_px",
        # --- pose columns, only filled when --pose is used ---
        # Blank means no person was found in that box at all, which is
        # itself informative: it usually means the box is on empty floor.
        f"fencer_{tag}_pose_found",
        f"fencer_{tag}_pose_conf",
        # Where the fencer's feet meet the floor. Much more meaningful
        # than a box edge, and the basis of the foot-height warning.
        f"fencer_{tag}_feet_x", f"fencer_{tag}_feet_y",
        f"fencer_{tag}_hip_x", f"fencer_{tag}_hip_y",
        f"fencer_{tag}_shoulder_x", f"fencer_{tag}_shoulder_y",
        # The extended wrist, which is the sword hand, guessed from geometry.
        f"fencer_{tag}_sword_wrist_x", f"fencer_{tag}_sword_wrist_y",
        f"fencer_{tag}_sword_elbow_x", f"fencer_{tag}_sword_elbow_y",
        # Reach from shoulders to sword hand, in torso-widths.
        f"fencer_{tag}_arm_extension",
        f"fencer_{tag}_feet_warning",
    ]


def _pose_cells(state):
    """The pose half of a row. All blank when --pose is off or no body found."""
    pose = state.get("pose")
    if pose is None:
        return ["", "", "", "", "", "", "", "", "", "", "", "", "",
                state.get("feet_warning", "")]
    wrist, elbow = pose.sword_hand()
    feet, hips, shoulders = pose.feet, pose.hips, pose.shoulders
    return [
        True, f"{pose.confidence:.4f}",
        f"{feet[0]:.1f}", f"{feet[1]:.1f}",
        f"{hips[0]:.1f}", f"{hips[1]:.1f}",
        f"{shoulders[0]:.1f}", f"{shoulders[1]:.1f}",
        f"{wrist[0]:.1f}", f"{wrist[1]:.1f}",
        f"{elbow[0]:.1f}", f"{elbow[1]:.1f}",
        f"{pose.arm_extension():.3f}",
        state.get("feet_warning", ""),
    ]


class TrackingCsv:
    """
    Writes one row per frame.

    When a fencer is lost, the position columns are left BLANK rather
    than repeating the last known box. A stale box that looks like a
    fresh measurement is exactly the kind of thing that makes you trust
    data you shouldn't. Blank means "we do not know".
    """

    def __init__(self, path: str):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(CSV_COLUMNS)

    def write_row(self, frame_index, timestamp_ms, camera, overlap, fencer_states):
        row = [
            frame_index,
            f"{timestamp_ms:.2f}",
            f"{camera['dx']:.3f}",
            f"{camera['dy']:.3f}",
            f"{camera['rotation_deg']:.4f}",
            f"{camera['scale']:.5f}",
            camera["reliable"],
            camera["points"],
            f"{overlap['iou']:.3f}",
            overlap["warning"],
        ]
        for state in fencer_states:
            if state["tracked"]:
                x, y, w, h = state["box"]
                row += [
                    True, x, y, w, h,
                    f"{state['center'][0]:.1f}", f"{state['center'][1]:.1f}",
                    f"{state['stable'][0]:.1f}", f"{state['stable'][1]:.1f}",
                    f"{state['speed']:.1f}",
                    state["movement"],
                    "" if state["score"] is None else f"{state['score']:.4f}",
                    f"{state['jump']:.1f}",
                ]
            else:
                row += [False, "", "", "", "", "", "", "", "", "",
                        state["movement"],
                        "" if state["score"] is None else f"{state['score']:.4f}",
                        ""]
            row += _pose_cells(state)
        self.writer.writerow(row)

    def close(self):
        self.file.close()
