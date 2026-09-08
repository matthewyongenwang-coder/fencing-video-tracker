"""
benchmark.py

Runs the tracker over a fixed set of real clips and reports how it did,
so that a change can be shown to help rather than assumed to.

    python3 benchmark.py              # run every case
    python3 benchmark.py sf seattle   # just those cases

WHY THIS EXISTS
Everything before this was tuned against one 4-second clip filmed from a
good angle with a steady camera. That is not what fencing footage looks
like. These cases are deliberately harder and more varied: five
competitions, four venues, fencers from 100 to 450 pixels tall, one clip
at 960x544, heavy motion blur, hand-held panning, and in one case a wall
of posters with pictures of fencers on them.

HOW IT IS SCORED
There is no automatic ground truth here. Nobody has labelled where
each fencer really is on every frame. So each run writes a filmstrip
image per case, and a human decides whether each box is still on the
right person at the end. That judgement gets recorded in EXPECTED below.

The numbers printed alongside (losses, overlap warnings, foot-height
warnings) are automatic, and are what you watch for regressions. They
are signals, not scores: a run with no warnings can still be wrong, and
a warning can be a false alarm.
"""

import argparse
import os
import sys

import cv2
import numpy as np

from tracking import Fencer
from camera import CameraMotion
from verify import OverlapWatch
from video_io import open_video

# Where the competition footage lives. Override without editing this file:
#   FENCING_VIDEOS="/path/to/your/clips" python3 benchmark.py
FENCING = os.environ.get(
    "FENCING_VIDEOS",
    os.path.expanduser("~/Documents/Fencing pics and vids"))

# name, video, first frame, how many frames, Fencer A box, Fencer B box
CASES = [
    ("portland-still",
     f"{FENCING}/Portland C2 comp 2026 Aug/IMG_1135 3.MOV",
     0, 120, (240, 430, 175, 215), (1470, 455, 230, 190)),

    ("portland-fleche",
     f"{FENCING}/Portland C2 comp 2026 Aug/IMG_1135 4.MOV",
     0, 221, (255, 425, 205, 240), (1490, 400, 200, 210)),

    # 960x544. Fencers only ~150px tall, piste at an angle, a spectator
    # in the foreground.
    ("seattle-lowres",
     f"{FENCING}/Seatle D2 2025/4834.MP4",
     210, 150, (118, 180, 65, 100), (515, 212, 70, 95)),

    # Severe motion blur, and the back wall is covered in large PHOTOS OF
    # FENCERS, about as unfair a set of distractors as exists.
    ("sf-blur-posters",
     f"{FENCING}/AFM SAN FRANSICO NOV 1 2025/Y14/2025-11-01-AFM-SemiFinal2.mp4",
     545, 150, (350, 400, 150, 210), (1075, 430, 100, 190)),

    # Fast hand-held pan plus extreme blur; fencers smeared across frames.
    ("cincy-pan-blur",
     f"{FENCING}/Febuary NAC at Cincinati/Junior/IMG_2274 2.MOV",
     1950, 150, (20, 255, 175, 205), (615, 290, 145, 145)),
]

# Filled in after looking at the filmstrips. True = the box was still on
# the right person at the end of the segment.
EXPECTED = {}

COLOR_A, COLOR_B = (0, 140, 255), (255, 90, 0)


def run_case(name, path, start, count, box_a, box_b, tracker="csrt",
             use_pose=False, use_detect=False, out_dir="_bench"):
    cap = open_video(path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, first = cap.read()
    if not ok:
        cap.release()
        return {"name": name, "error": "could not read the first frame"}

    fencers = [Fencer("Fencer A", COLOR_A, first, box_a, kind=tracker),
               Fencer("Fencer B", COLOR_B, first, box_b, kind=tracker)]
    camera = CameraMotion(first)
    overlap = OverlapWatch()

    gate = None
    if use_detect:
        from detect import DetectionGate
        gate = DetectionGate()
    snaps = 0

    estimator = foot_watches = None
    if use_pose:
        from pose import PoseEstimator, FootWatch
        estimator = PoseEstimator()
        foot_watches = [FootWatch(), FootWatch()]

    losses = [0, 0]
    no_body = [0, 0]
    declared_lost = [False, False]
    recoveries = [0, 0]
    frames_lost = [0, 0]
    GIVE_UP_AFTER = 12          # 0.4 seconds with nobody in the box
    overlap_warnings = 0
    foot_warnings = [0, 0]
    strip_at = {0, count // 3, (2 * count) // 3, count - 1}
    strips = []

    frame = first
    for i in range(count):
        if i > 0:
            ok, frame = cap.read()
            if not ok:
                break

        camera.update(frame, [f.box for f in fencers])
        for j, fencer in enumerate(fencers):
            tracked, _ = fencer.update(frame)
            if not tracked:
                losses[j] += 1

        if gate is not None:
            for j, info in enumerate(gate.step(frame, fencers, camera, 1/30.0)):
                if info["snapped"]:
                    snaps += 1
                if info["newly_lost"]:
                    declared_lost[j] = True
                if info["recovered"]:
                    recoveries[j] += 1

        for j in range(2):
            if not fencers[j].tracked:
                frames_lost[j] += 1

        _iou, _warn, new_overlap = overlap.update(
            fencers[0].box, fencers[1].box,
            fencers[0].tracked and fencers[1].tracked)
        if new_overlap:
            overlap_warnings += 1

        if estimator is not None:
            for j, fencer in enumerate(fencers):
                result = estimator.estimate(frame, fencer.box) \
                    if fencer.tracked else None
                _y, _w, new_foot = foot_watches[j].update(result)
                if new_foot:
                    foot_warnings[j] += 1

        if i in strip_at:
            shot = frame.copy()
            for fencer in fencers:
                x, y, w, h = fencer.box
                colour = fencer.color if fencer.tracked else (0, 0, 255)
                cv2.rectangle(shot, (x, y), (x + w, y + h), colour, 3)
                cv2.putText(shot, fencer.label[-1], (x, max(18, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3)
            cv2.putText(shot, f"{name} +{i}", (16, shot.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            strips.append(cv2.resize(shot, (760, int(760 * shot.shape[0] /
                                                     shot.shape[1]))))
    cap.release()

    os.makedirs(out_dir, exist_ok=True)
    if strips:
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), np.vstack(strips))

    return {
        "name": name,
        "frames": count,
        "losses": losses,
        "overlap_warnings": overlap_warnings,
        "foot_warnings": foot_warnings,
        "final": [f.box for f in fencers],
        "snaps": snaps,
        "frames_lost": frames_lost,
        "recoveries": recoveries,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the tracking benchmark.")
    parser.add_argument("cases", nargs="*", help="Case names, or none for all")
    parser.add_argument("--tracker", choices=["csrt", "vit"], default="csrt")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--detect", action="store_true",
                        help="Snap boxes to detected people (see detect.py)")
    parser.add_argument("--out-dir", default="_bench")
    args = parser.parse_args()

    missing = [c[0] for c in CASES if not os.path.isfile(c[1])]
    if missing:
        print(f"Clips not found for: {', '.join(missing)}")
        print(f"Looked under: {FENCING}")
        print("Set FENCING_VIDEOS to point at your footage, or edit CASES.\n")

    chosen = [c for c in CASES
              if (not args.cases or c[0] in args.cases) and os.path.isfile(c[1])]
    if not chosen:
        print("No matching cases. Available:",
              ", ".join(c[0] for c in CASES))
        sys.exit(2)

    print(f"tracker={args.tracker}  pose={'on' if args.pose else 'off'}  "
          f"detect={'on' if args.detect else 'off'}\n")
    header = (f"{'case':18} {'frames':>6} {'lost A/B':>10} {'overlap':>8} "
              f"{'feet A/B':>9} {'snaps':>6} {'no-body A/B':>12}")
    print(header)
    print("-" * len(header))

    for case in chosen:
        result = run_case(*case, tracker=args.tracker, use_pose=args.pose,
                          use_detect=args.detect, out_dir=args.out_dir)
        if "error" in result:
            print(f"{result['name']:18} ERROR: {result['error']}")
            continue
        print(f"{result['name']:18} {result['frames']:6} "
              f"{result['losses'][0]:4}/{result['losses'][1]:<5} "
              f"{result['overlap_warnings']:8} "
              f"{result['foot_warnings'][0]:4}/{result['foot_warnings'][1]:<4} "
              f"{result.get('snaps', 0):6} "
              f"{result.get('frames_lost',[0,0])[0]:5}/"
              f"{result.get('frames_lost',[0,0])[1]:<6}")

    print(f"\nFilmstrips written to {args.out_dir}/. Look at them. "
          "Warning counts alone do not tell you whether the boxes are on "
          "the right people.")


if __name__ == "__main__":
    main()
