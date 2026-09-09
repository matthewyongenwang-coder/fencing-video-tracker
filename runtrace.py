"""
runtrace.py

A machine-checkable record of exactly what the tracker did on one clip.

WHY THIS EXISTS
benchmark.py scores by eye. I run five clips, look at the filmstrips, and
decide whether each box is still on the right person. That is honest, and
it is the only real ground truth I have, but it only works when I am
tuning one number in one Python file.

It stops working the moment I move this engine anywhere else. A port
changes thousands of small numbers at once, and I am not going to eyeball
five filmstrips after every commit for four months. Without something
automatic, every regression during that work is silent.

So: one JSON file per run, holding every number the run produced, plus
enough provenance to know whether two runs are even comparable.
compare_traces.py diffs two of them and exits non-zero when something
moved.

WHAT A TRACE IS FOR, AND WHAT IT IS NOT FOR
A trace proves two runs behaved the same. It does not prove either run was
correct. The box can be confidently tracking a referee in both, and the
trace will report perfect agreement. That judgement stays with EXPECTED in
benchmark.py and with my own eyes on the filmstrip. Trace agreement plus an
unchanged EXPECTED verdict is the pair that means something. Neither one
alone does.

WHY THE HEADER MATTERS AS MUCH AS THE FRAMES
Two runs that disagree because they used different model files, a different
OpenCV, or a different start frame are not a regression, they are a mistake
in how I ran them. The header records enough to catch that before I go
looking for a bug that is not there. Model files are hashed rather than
named, because a file called yolox_person.onnx is not proof of which
yolox_person.onnx.

WHY EVERY FRAME AND NOT A SUMMARY
I thought about storing counts of losses and warnings instead. benchmark.py
already prints those, and they are exactly what hides a problem: a box can
slide onto the wrong person without changing any count. What I need is the
first frame where two runs disagree, and you cannot recover that from a
total.
"""

import hashlib
import io
import json
import os
import platform

import cv2

TRACE_FORMAT = 1


def _file_sha256(path):
    """Short hash of a file, or None when it is not there."""
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def model_hashes(models_dir=None):
    """Hash whichever pretrained models are on disk."""
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models")
    names = ("yolox_person.onnx", "pose_landmarks.onnx", "vittrack.onnx")
    found = {}
    for name in names:
        digest = _file_sha256(os.path.join(models_dir, name))
        if digest is not None:
            found[name] = digest
    return found


def _round(value, places):
    """
    Round for storage, so a trace does not carry float noise it cannot
    justify. Positions are already sub-pixel; nothing here needs 17
    significant figures, and writing them out invites a diff over a last
    bit that no measurement supports.
    """
    if value is None:
        return None
    return round(float(value), places)


class TraceWriter:
    """
    Collects one record per frame and writes the whole file at the end.

    Held in memory on purpose. A capped benchmark run is 150 to 221 frames,
    so this stays well under a megabyte, and writing once means a trace is
    either complete or absent. A half-written trace that still parses is
    worse than no trace, because it compares clean against the frames it
    happens to have.
    """

    def __init__(self, path, header):
        self.path = path
        self.header = dict(header)
        self.header["trace_format"] = TRACE_FORMAT
        self.header["opencv_version"] = cv2.__version__
        self.header["python_version"] = platform.python_version()
        self.header["platform"] = f"{platform.system()}-{platform.machine()}"
        self.frames = []
        self.events = []

    def add_event(self, kind, frame_index, time_sec, who=None, detail=None):
        """
        A LOST, RECOVERED, OVERLAP, FEET or SNAP moment.

        Recorded separately from the frame rows because these are the lines
        I actually read. A loss that moves by three frames is a real change
        in behaviour, and it is easy to miss inside 221 rows of
        mostly-identical numbers.
        """
        self.events.append({
            "kind": kind,
            "frame": int(frame_index),
            "t_sec": _round(time_sec, 4),
            "who": who,
            "detail": detail,
        })

    def add_frame(self, frame_index, timestamp_ms, camera, overlap, states):
        self.frames.append({
            "frame": int(frame_index),
            "t_ms": _round(timestamp_ms, 2),
            "camera": {
                "dx": _round(camera["dx"], 3),
                "dy": _round(camera["dy"], 3),
                "rotation_deg": _round(camera["rotation_deg"], 4),
                "scale": _round(camera["scale"], 5),
                "reliable": bool(camera["reliable"]),
                "points": int(camera["points"]),
            },
            "overlap": {
                "iou": _round(overlap["iou"], 3),
                "warning": bool(overlap["warning"]),
            },
            "fencers": [self._fencer_row(state) for state in states],
        })

    def _fencer_row(self, state):
        row = {
            "tracked": bool(state["tracked"]),
            "movement": state["movement"],
            "detected": state["detected"],
            "feet_warning": bool(state["feet_warning"]),
            "score": _round(state["score"], 4),
        }
        # A lost fencer has no position. Blank, not the last known box, for
        # the same reason TrackingCsv leaves those cells empty: a stale
        # number that looks like a fresh measurement is how you end up
        # trusting data you should not.
        if state["tracked"]:
            row["box"] = [int(v) for v in state["box"]]
            row["center"] = [_round(state["center"][0], 1),
                             _round(state["center"][1], 1)]
            row["stable"] = [_round(state["stable"][0], 1),
                             _round(state["stable"][1], 1)]
            row["speed_px_s"] = _round(state["speed"], 1)
            row["jump_px"] = _round(state["jump"], 1)
        else:
            row["box"] = None
            row["center"] = None
            row["stable"] = None
            row["speed_px_s"] = None
            row["jump_px"] = None

        pose = state.get("pose")
        if pose is None:
            row["pose"] = None
        else:
            wrist, elbow = pose.sword_hand()
            row["pose"] = {
                "confidence": _round(pose.confidence, 4),
                "feet": [_round(pose.feet[0], 1), _round(pose.feet[1], 1)],
                "hips": [_round(pose.hips[0], 1), _round(pose.hips[1], 1)],
                "shoulders": [_round(pose.shoulders[0], 1),
                              _round(pose.shoulders[1], 1)],
                "sword_wrist": [_round(wrist[0], 1), _round(wrist[1], 1)],
                "sword_elbow": [_round(elbow[0], 1), _round(elbow[1], 1)],
                "arm_extension": _round(pose.arm_extension(), 3),
            }
        return row

    def close(self):
        payload = {
            "header": self.header,
            "events": self.events,
            "frames": self.frames,
        }
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        # sort_keys so the bytes do not depend on dict insertion order, and
        # a trailing newline so git treats it as an ordinary text file.
        with io.open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.write("\n")
        return self.path


def load(path):
    with io.open(path, encoding="utf-8") as handle:
        return json.load(handle)
