"""
video_io.py

Opening a video file and reading frames with their REAL timestamps.

Nothing in here knows about fencers or tracking -- it just turns a file
path into frames + times. Keeping it separate means that when you move
to the iPhone app later, this is the only file that has to be replaced.
"""

import os
import cv2


def open_video(path: str) -> cv2.VideoCapture:
    """
    Open a video file for reading.

    Filenames with spaces: inside Python a path with spaces is just a
    normal string, so "IMG_1135 3.MOV" works with no escaping. The only
    place spaces matter is when you type the path into Terminal, where
    you must wrap it in quotes. expanduser() makes "~/..." work too.
    """
    full_path = os.path.abspath(os.path.expanduser(path))

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Video file not found: {full_path}")

    cap = cv2.VideoCapture(full_path)

    if not cap.isOpened():
        raise IOError(
            f"OpenCV could not open this video: {full_path}\n"
            "Some iPhone HEVC / Dolby Vision clips need an OpenCV build with "
            "FFMPEG support. Check `python3 -c \"import cv2; "
            "print(cv2.getBuildInformation())\"` for a FFMPEG: YES line. "
            "As a workaround, re-export the clip as H.264 .mp4 and retry."
        )

    return cap


def measure_timebase(path: str) -> dict:
    """
    Decode the clip once, quickly, just to collect each frame's real
    timestamp. Returns the true frame count and the true average fps.

    WHY THIS EXISTS -- this is a real bug fix, not decoration:

    cv2.CAP_PROP_FPS and cv2.CAP_PROP_FRAME_COUNT come from the file's
    metadata header, and on this iPhone clip BOTH are wrong:
        metadata says  127 frames @ 28.20 fps
        actually       120 frames @ 29.97 fps
    Feeding the metadata fps into the output VideoWriter made the
    exported video 6.3% too slow (4.256s instead of 4.003s). Measuring
    the timestamps ourselves fixes that.

    This uses cap.grab() instead of cap.read(). grab() decodes the frame
    but skips converting it into a numpy array we don't need, so this
    whole pre-pass takes about 0.3s on a 4 second clip.
    """
    cap = open_video(path)
    timestamps_ms = []
    while cap.grab():
        timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()

    if len(timestamps_ms) < 2:
        raise IOError(f"Only {len(timestamps_ms)} frame(s) decoded from {path}")

    gaps = [timestamps_ms[i + 1] - timestamps_ms[i]
            for i in range(len(timestamps_ms) - 1)]
    mean_gap_ms = sum(gaps) / len(gaps)

    return {
        "frame_count": len(timestamps_ms),
        "measured_fps": 1000.0 / mean_gap_ms,
        "duration_sec": (timestamps_ms[-1] + mean_gap_ms) / 1000.0,
        "min_gap_ms": min(gaps),
        "max_gap_ms": max(gaps),
        # How uneven the spacing is. ~0 means constant frame rate.
        "gap_spread_ms": max(gaps) - min(gaps),
    }


def get_video_info(cap: cv2.VideoCapture, path: str) -> dict:
    """
    Size plus both versions of the timing: what the file's header claims,
    and what we actually measured. main.py prints both so a mismatch is
    visible rather than silently wrong.
    """
    measured = measure_timebase(path)
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "metadata_fps": cap.get(cv2.CAP_PROP_FPS),
        "metadata_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        **measured,
    }


def seek_to(cap: cv2.VideoCapture, index: int):
    """
    Jump to a frame number. Needed because your real clips are whole
    bouts -- the interesting four seconds are somewhere in the middle of
    seventy, and nobody wants to track the standing-around.

    Seeking in HEVC is not always exact: the decoder can only jump to a
    keyframe and then roll forward, so the frame you land on may be a
    frame or two off. That is fine for choosing where to start, and it is
    why the CSV records real timestamps rather than counting frames.
    """
    if index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    return cap


def read_frame_at(path: str, index: int):
    """Grab one frame by number, for previewing. Returns None if past the end."""
    cap = open_video(path)
    seek_to(cap, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def read_first_frame(cap: cv2.VideoCapture):
    """
    Read frame 0 for the selection step, then rewind so the tracking loop
    still starts at frame 0.

    Verified on this clip: after the rewind, cap.read() returns a frame
    that is byte-for-byte identical to the first one, and the loop still
    yields all 120 frames. (Seeking in HEVC is not always this clean, so
    this was checked rather than assumed.)
    """
    position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ok, frame = cap.read()
    if not ok:
        raise IOError("Could not read the first frame of the video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, position)
    return frame


def frame_iterator(cap: cv2.VideoCapture):
    """
    Yield (frame_index, timestamp_ms, frame) for every frame.

    Note the ORDER here: cap.get(CAP_PROP_POS_MSEC) is called AFTER
    cap.read(). With the FFMPEG backend this returns the timestamp of the
    frame that was just read, which is what we want. Checked directly on
    this clip: the values come out as 0.00, 33.33, 66.67, ... so frame 0
    is correctly stamped t=0 rather than being shifted by one frame.
    """
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        yield frame_index, timestamp_ms, frame
        frame_index += 1
