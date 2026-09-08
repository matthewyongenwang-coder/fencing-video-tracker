"""
main.py

Fencing video tracker.

Pipeline:
    load video  ->  you box each fencer  ->  track both through the clip
                ->  live window + annotated video + CSV

Run it:
    python3 main.py "/path/to/IMG_1135 3.MOV"

While it's running:
    SPACE   pause / resume
    A       pause and re-select Fencer A
    B       pause and re-select Fencer B
    Q, ESC  stop early (whatever has been processed is still saved)

If a tracker reports a loss, the video pauses by itself and tells you
which fencer to re-select. Nothing is silently patched over.
"""

import argparse
import os
import sys

import cv2

from video_io import (open_video, get_video_info, read_first_frame,
                      frame_iterator, seek_to)
from tracking import Fencer
from camera import CameraMotion
from motion import PisteAxis, MovementLabeler
from verify import OverlapWatch
from pose import PoseEstimator, FootWatch, SKELETON
from detect import DetectionGate
from export import open_writer, verify_output, TrackingCsv

LABEL_A = "Fencer A"
LABEL_B = "Fencer B"
COLOR_A = (0, 140, 255)    # orange, BGR
COLOR_B = (255, 90, 0)     # blue, BGR
COLOR_LOST = (0, 0, 255)   # red
COLOR_HUD = (255, 255, 255)


# ----------------------------------------------------------------------
# Selecting boxes
# ----------------------------------------------------------------------

def _scale_for_display(frame, max_width):
    """
    Shrink a frame so it fits on a laptop screen, and return the factor
    needed to convert coordinates back to full resolution.

    This matters more than it looks. The clip is 1920x1080. On a MacBook
    screen an unscaled OpenCV window is bigger than the display, so the
    bottom of the piste is off-screen and you cannot see what you are
    boxing. Everything drawn is scaled for viewing only. All tracking,
    all CSV numbers, and the exported video stay at full 1920x1080.
    """
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    scale = max_width / float(width)
    small = cv2.resize(frame, (int(width * scale), int(height * scale)),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def select_box(frame, window_title, max_width, hint):
    """
    Interactive click-and-drag selection, on a screen-sized copy of the
    frame, with the box converted back to full-resolution pixels.

    Returns None if you cancel with 'c'.
    """
    small, scale = _scale_for_display(frame, max_width)
    canvas = small.copy()
    cv2.putText(canvas, hint, (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    cv2.putText(canvas, "drag a box, then ENTER or SPACE   (c = cancel)",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    box = cv2.selectROI(window_title, canvas, showCrosshair=True,
                        fromCenter=False, printNotice=False)
    cv2.destroyWindow(window_title)
    cv2.waitKey(1)   # let the Cocoa window actually go away

    if box is None or box[2] == 0 or box[3] == 0:
        return None

    inv = 1.0 / scale
    return (int(box[0] * inv), int(box[1] * inv),
            int(box[2] * inv), int(box[3] * inv))


def select_box_or_exit(frame, title, max_width, hint):
    box = select_box(frame, title, max_width, hint)
    if box is None:
        raise ValueError(
            f"No box drawn for {title}. Nothing to track, so stopping. "
            "Re-run and drag a box around the fencer.")
    return box


def parse_box(text):
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("Box must be given as x,y,w,h")
    return tuple(parts)


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------

def make_state(fencer, tracked, center, stable, speed, movement):
    return {"tracked": tracked, "box": fencer.box, "center": center,
            "stable": stable, "speed": speed, "movement": movement,
            "jump": fencer.last_jump_px, "score": fencer.score,
            "pose": None, "feet_warning": "", "detected": ""}


# How tall a box has to be, relative to its width, before we suspect you
# boxed the whole fencer instead of just the torso. A torso+mask box is
# roughly square-ish (about 1.2); a full-body box is about 1.7.
FULL_BODY_ASPECT = 1.5

TORSO_ADVICE = (
    "TIP: box the MASK AND TORSO ONLY, not the legs and not the blade.\n"
    "     Measured on your own clips: a full-body box drifted onto the\n"
    "     referee after 49 frames, while a torso box on the same clip\n"
    "     followed the fencer correctly the whole way. Legs swing around\n"
    "     and a tall box is mostly empty floor, which is what the tracker\n"
    "     ends up learning."
)


def warn_if_full_body(label, box):
    """Say something if the drawn box looks like a whole fencer."""
    x, y, w, h = box
    if w > 0 and h / float(w) > FULL_BODY_ASPECT:
        print(f"  NOTE: {label}'s box is tall and narrow ({w}x{h}), which "
              f"looks like a full-body box.")
        print("        A torso+mask box usually tracks much better here.")


def draw_skeleton(frame, pose_result, color):
    """The stick figure: joints as dots, limbs as lines."""
    points = pose_result.landmarks
    for a, b in SKELETON:
        pa = tuple(int(v) for v in points[a])
        pb = tuple(int(v) for v in points[b])
        cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(points):
        # Dim the points the model itself is unsure about.
        bright = pose_result.visibility is None or pose_result.visibility[i] > 0.5
        cv2.circle(frame, (int(x), int(y)), 4 if bright else 2,
                   (255, 255, 255) if bright else (140, 140, 140), -1)
    # Mark the two things we actually use downstream.
    feet = pose_result.feet
    cv2.circle(frame, (int(feet[0]), int(feet[1])), 8, (0, 255, 255), 2)
    wrist, _elbow = pose_result.sword_hand()
    cv2.circle(frame, (int(wrist[0]), int(wrist[1])), 8, (0, 0, 255), 2)


def draw_overlay(frame, fencers, states, frame_index, time_sec, camera,
                 paused, overlap=None):
    for fencer, state in zip(fencers, states):
        if state.get("pose") is not None:
            draw_skeleton(frame, state["pose"], fencer.color)
        x, y, w, h = fencer.box
        if state["tracked"]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), fencer.color, 3)
            text = f"{fencer.label}: {state['movement']}"
            cv2.putText(frame, text, (x, max(24, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, fencer.color, 2)
        else:
            # Dashed-looking stale box in red, so a lost fencer is
            # obvious rather than just missing.
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_LOST, 2)
            cv2.putText(frame, f"{fencer.label}: LOST", (x, max(24, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_LOST, 2)

    hud = [
        f"frame {frame_index}   t={time_sec:5.2f}s",
        ("camera: dx={:+.1f} dy={:+.1f} px  rot={:+.3f}deg  zoom={:.4f}".format(
            camera["dx"], camera["dy"], camera["rotation_deg"], camera["scale"])
         if camera["reliable"] else "camera: NOT MEASURED (too few points)"),
    ]
    for i, line in enumerate(hud):
        cv2.putText(frame, line, (20, 40 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, COLOR_HUD, 2)

    for fencer, state in zip(fencers, states):
        if state.get("feet_warning"):
            cv2.putText(frame, f"{fencer.label}: FEET TOO HIGH - probably "
                        f"on someone in the background",
                        (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_LOST, 2)

    if overlap and overlap["warning"]:
        cv2.putText(frame,
                    "BOTH BOXES ON THE SAME PERSON - re-select (A / B)",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_LOST, 3)

    if paused:
        cv2.putText(frame, "PAUSED  -  SPACE resume, A/B re-select, Q quit",
                    (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Track two hand-selected fencers through a clip.")
    parser.add_argument("video_path", help="Path to the .MOV / .mp4 clip")
    parser.add_argument("--box-a", default=None,
                        help="x,y,w,h for Fencer A, skips the picker")
    parser.add_argument("--box-b", default=None,
                        help="x,y,w,h for Fencer B, skips the picker")
    parser.add_argument("--out-dir", default="output",
                        help="Where to write the video + CSV")
    parser.add_argument("--no-display", action="store_true",
                        help="Process without the live window (for batch runs)")
    parser.add_argument("--display-width", type=int, default=1280,
                        help="Width of the on-screen window in pixels")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Frame to begin at. Clips are usually whole bouts; "
                             "the interesting part is rarely at the start.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="How many frames to process, 0 for all")
    parser.add_argument("--progress", action="store_true",
                        help="Print machine-readable PROGRESS lines "
                             "(used by the web page)")
    parser.add_argument("--detect", action="store_true",
                        help="Keep each box locked to a DETECTED person, and "
                             "declare a fencer lost when no body is in their "
                             "box. Strongly recommended on real footage.")
    parser.add_argument("--pose", action="store_true",
                        help="Also find body landmarks (feet, wrists, joints). "
                             "Roughly halves speed. Draws a stick figure and "
                             "adds pose columns to the CSV.")
    parser.add_argument("--tracker", choices=["csrt", "vit"], default="csrt",
                        help="csrt (default, sticky but fails silently) or "
                             "vit (gives up sooner but reports a real score)")
    args = parser.parse_args()

    show = not args.no_display

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video_path))[0]
    out_video = os.path.join(args.out_dir, f"{base}_tracked.mp4")
    out_csv = os.path.join(args.out_dir, f"{base}_tracking.csv")

    # --- load -----------------------------------------------------------
    cap = open_video(args.video_path)
    info = get_video_info(cap, args.video_path)

    print(f"Loaded: {info['width']}x{info['height']}")
    print(f"  file header claims : {info['metadata_frame_count']} frames @ "
          f"{info['metadata_fps']:.2f} fps")
    print(f"  actually measured  : {info['frame_count']} frames @ "
          f"{info['measured_fps']:.2f} fps  ({info['duration_sec']:.3f}s)")
    if info["metadata_frame_count"] != info["frame_count"] or \
            abs(info["metadata_fps"] - info["measured_fps"]) > 0.5:
        print("  ^ header and reality disagree. Using the measured values.")
    print(f"  frame gaps: {info['min_gap_ms']:.2f}-{info['max_gap_ms']:.2f} ms "
          f"(spread {info['gap_spread_ms']:.2f} ms)")

    seek_to(cap, args.start_frame)
    first_frame = read_first_frame(cap)
    total_to_do = info["frame_count"] - args.start_frame
    if args.max_frames:
        total_to_do = min(total_to_do, args.max_frames)

    # --- select ---------------------------------------------------------
    if args.box_a:
        box_a = parse_box(args.box_a)
    else:
        print("\n" + TORSO_ADVICE)
        print("\nDraw a box around FENCER A's mask and torso.")
        box_a = select_box_or_exit(first_frame, "Select Fencer A",
                                   args.display_width,
                                   "FENCER A - mask + torso only")
    if args.box_b:
        box_b = parse_box(args.box_b)
    else:
        print("Now draw a box around FENCER B's mask and torso.")
        box_b = select_box_or_exit(first_frame, "Select Fencer B",
                                   args.display_width,
                                   "FENCER B - mask + torso only")

    print(f"\nFencer A box: {box_a}")
    print(f"Fencer B box: {box_b}")
    warn_if_full_body(LABEL_A, box_a)
    warn_if_full_body(LABEL_B, box_b)

    fencer_a = Fencer(LABEL_A, COLOR_A, first_frame, box_a, kind=args.tracker)
    fencer_b = Fencer(LABEL_B, COLOR_B, first_frame, box_b, kind=args.tracker)
    fencers = [fencer_a, fencer_b]

    # Piste direction, measured from where you actually put the boxes.
    axis = PisteAxis(fencer_a.center(), fencer_b.center())
    print(axis.describe())
    labelers = [MovementLabeler(axis, sign=+1), MovementLabeler(axis, sign=-1)]

    camera_tracker = CameraMotion(first_frame)
    overlap_watch = OverlapWatch()

    gate = None
    if args.detect:
        gate = DetectionGate()
        print("Person detection ON. Boxes will snap to detected bodies, and a "
              "fencer with no body in their box gets declared lost.")

    pose_estimator = None
    foot_watches = [None, None]
    if args.pose:
        pose_estimator = PoseEstimator()
        foot_watches = [FootWatch(), FootWatch()]
        print("Pose estimation ON (this roughly halves processing speed).")

    # --- outputs --------------------------------------------------------
    writer, codec = open_writer(out_video, info["measured_fps"],
                                (info["width"], info["height"]))
    print(f"\nWriting video with codec '{codec}' at "
          f"{info['measured_fps']:.2f} fps")
    csv_out = TrackingCsv(out_csv)

    window = "Fencing tracker"
    if show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, args.display_width,
                         int(args.display_width * info["height"] / info["width"]))
        print("Controls: SPACE pause | A / B re-select | Q quit\n")

    paused = False
    quit_early = False
    frames_written = 0
    loss_events = []
    overlap_events = []
    feet_events = []

    previous_time = None
    for frame_index, timestamp_ms, frame in frame_iterator(cap):
        time_sec = timestamp_ms / 1000.0
        # Real gap between frames, for the motion prediction. Using the
        # measured timestamps rather than assuming 1/30 keeps the
        # velocities honest on variable-rate footage.
        dt = (1.0 / info["measured_fps"]) if previous_time is None \
            else max(time_sec - previous_time, 1e-3)
        previous_time = time_sec

        # 1. How much did the camera move? Mask out the fencers so their
        #    own motion doesn't get mistaken for camera shake.
        camera = camera_tracker.update(frame, [f.box for f in fencers])

        # 2. Update both trackers.
        states = []
        for fencer, labeler in zip(fencers, labelers):
            tracked, _box = fencer.update(frame)
            center = fencer.center()
            # Undo the camera's slide, rotation and zoom, so this position
            # is "where the fencer would be if the camera never moved".
            stable = camera_tracker.stabilise(center)
            movement, speed = labeler.update(time_sec, stable, tracked)
            states.append(make_state(fencer, tracked, center, stable,
                                     speed, movement))

        # 2b. Body landmarks, if asked for. Pointed at each fencer's own
        #     box, so the skeleton always belongs to the fencer YOU chose.
        if pose_estimator is not None:
            for idx, (fencer, state) in enumerate(zip(fencers, states)):
                result = pose_estimator.estimate(frame, fencer.box) \
                    if state["tracked"] else None
                state["pose"] = result
                feet_y, feet_warning, new_feet = foot_watches[idx].update(result)
                state["feet_warning"] = feet_warning
                if new_feet:
                    print(f"[frame {frame_index}, t={time_sec:.2f}s] "
                          f"WARNING: {fencer.label}'s feet have jumped "
                          f"{foot_watches[idx].reference - feet_y:.0f}px up the "
                          f"frame. That usually means the box has wandered "
                          f"onto someone in the background.")
                    feet_events.append((frame_index, time_sec, fencer.label))
                    if show:
                        paused = True
                        print("       Paused. Re-select with A or B.")

        # 2c. Keep each box on an actual person, and admit it when we
        #     cannot find one. See detect.py for why this matters most.
        if gate is not None:
            gate_report = gate.step(frame, fencers, camera_tracker, dt)
            for idx, (fencer, info) in enumerate(zip(fencers, gate_report)):
                states[idx]["box"] = fencer.box
                states[idx]["tracked"] = fencer.tracked
                states[idx]["detected"] = info["matched"]
                if info["snapped"]:
                    labelers[idx].reset()
                    if foot_watches[idx] is not None:
                        foot_watches[idx].reset()
                if info["newly_lost"]:
                    print(f"[frame {frame_index}, t={time_sec:.2f}s] "
                          f"{fencer.label} LOST - no person found in the box "
                          f"({info['people_seen']} people detected in frame). "
                          f"The box was probably sliding onto scenery.")
                    loss_events.append((frame_index, time_sec, fencer.label))
                    if show:
                        paused = True
                        print("       Paused. Re-select with A or B.")
                elif info["recovered"]:
                    print(f"[frame {frame_index}, t={time_sec:.2f}s] "
                          f"{fencer.label} re-attached to a detected person. "
                          f"Check it is the RIGHT person.")

        # 3. Two fencers cannot be in the same place. If both boxes have
        #    sat on top of each other for a while, at least one tracker is
        #    on the wrong person.
        iou, overlap_warning, new_overlap = overlap_watch.update(
            fencer_a.box, fencer_b.box,
            states[0]["tracked"] and states[1]["tracked"])
        overlap = {"iou": iou, "warning": overlap_warning}
        if new_overlap:
            print(f"[frame {frame_index}, t={time_sec:.2f}s] "
                  f"WARNING: both boxes are on the same spot (overlap "
                  f"{iou:.0%}). At least one tracker is on the wrong person.")
            overlap_events.append((frame_index, time_sec))
            if show:
                paused = True
                print("       Paused. Press A and/or B to re-select.")

        # 4. Report losses loudly, and stop for you to fix them.
        for fencer, state in zip(fencers, states):
            if not state["tracked"] and fencer.frames_lost == 1:
                print(f"[frame {frame_index}, t={time_sec:.2f}s] {fencer.label} LOST")
                loss_events.append((frame_index, time_sec, fencer.label))
                if show:
                    paused = True
                    print(f"       Paused. Press "
                          f"{'A' if fencer.label == LABEL_A else 'B'} to "
                          f"re-select {fencer.label}, or SPACE to carry on.")

        # 5. Show the frame and handle your keypresses.
        #
        # This happens BEFORE the frame is written, so that if you
        # re-select a fencer here, the corrected box lands in the saved
        # video and CSV for THIS frame, not starting from the next one.
        def show_now():
            preview = frame.copy()
            draw_overlay(preview, fencers, states, frame_index, time_sec,
                         camera, paused, overlap)
            small, _ = _scale_for_display(preview, args.display_width)
            cv2.imshow(window, small)

        if show:
            show_now()
            while True:
                key = cv2.waitKey(0 if paused else 1) & 0xFF

                if key in (ord('q'), ord('Q'), 27):
                    quit_early = True
                    break

                if key == ord(' '):
                    paused = not paused
                    show_now()
                    if not paused:
                        break
                    continue

                if key in (ord('a'), ord('A'), ord('b'), ord('B')):
                    which = 0 if key in (ord('a'), ord('A')) else 1
                    target, labeler = fencers[which], labelers[which]
                    was_paused, paused = paused, True
                    print(f"  re-selecting {target.label} at frame {frame_index}...")
                    new_box = select_box(
                        frame, f"Re-select {target.label}", args.display_width,
                        f"RE-SELECT {target.label.upper()}  (frame {frame_index})")
                    if new_box is None:
                        print("  Cancelled, nothing changed.")
                        paused = was_paused
                    else:
                        target.reinit(frame, new_box)
                        if foot_watches[which] is not None:
                            foot_watches[which].reset()
                        if gate is not None:
                            gate.reset(target)
                        # Wipe the speed history: comparing a fresh
                        # hand-drawn box against positions from before
                        # the loss would invent a movement that never
                        # happened. Speed restarts from zero.
                        labeler.reset()
                        center = target.center()
                        stable = camera_tracker.stabilise(center)
                        labeler.update(time_sec, stable, True)
                        states[which] = make_state(
                            target, True, center, stable, 0.0, "Still")
                        print(f"  {target.label} re-seeded at {new_box}")
                    show_now()
                    continue

                if not paused:
                    break

        # 6. Save this frame, with whatever corrections you just made.
        annotated = frame.copy()
        draw_overlay(annotated, fencers, states, frame_index, time_sec,
                     camera, paused, overlap)
        writer.write(annotated)
        frames_written += 1
        if args.progress:
            print(f"PROGRESS {frames_written} {total_to_do}", flush=True)
        if args.max_frames and frames_written >= args.max_frames:
            break
        csv_out.write_row(frame_index, timestamp_ms, camera, overlap, states)

        if quit_early:
            print(f"\nStopped early at frame {frame_index}.")
            break

    # --- close up -------------------------------------------------------
    cap.release()
    writer.release()
    csv_out.close()
    if show:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    print(f"\nProcessed {frames_written} frames.")

    check = verify_output(out_video, frames_written)
    if check["ok"]:
        print(f"Video verified: {check['frames_decoded']} frames decoded back, "
              f"{check['duration_sec']:.3f}s, "
              f"{check['size_bytes'] / 1_000_000:.1f} MB")
    else:
        print(f"VIDEO PROBLEM: {check['reason']}")

    print(f"  video: {out_video}")
    print(f"  csv:   {out_csv}")

    if loss_events:
        print(f"\n{len(loss_events)} tracking loss event(s):")
        for idx, t, label in loss_events:
            print(f"  frame {idx} (t={t:.2f}s): {label}")
    else:
        print("\nNo tracking losses were reported.")
        print("Note: 'no loss reported' is not the same as 'always on the right "
              "person'. CSRT reports success even if it has drifted onto "
              "someone else. Watch the exported video to confirm.")

    if overlap_events:
        print(f"\n{len(overlap_events)} overlap warning(s), both boxes on "
              f"one person:")
        for idx, t in overlap_events:
            print(f"  frame {idx} (t={t:.2f}s)")

    if feet_events:
        print(f"\n{len(feet_events)} foot-height warning(s):")
        for idx, t, label in feet_events:
            print(f"  frame {idx} (t={t:.2f}s): {label}")

    reinits = sum(f.reinit_count for f in fencers)
    if reinits:
        print(f"You re-selected a fencer {reinits} time(s) during the run.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
