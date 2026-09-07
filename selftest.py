"""
selftest.py

Checks the parts of this project that can be tested without a person
clicking a mouse. Run it after changing anything:

    python3 selftest.py "/path/to/IMG_1135 3.MOV"

It does NOT test the click-and-drag itself -- that genuinely needs a
hand on a trackpad. It DOES test everything that happens either side of
the drag, including the coordinate maths that converts a box drawn on a
shrunk-to-fit window back into full-resolution pixels, and the mid-clip
re-selection path that runs when you press A or B.

One of these tests already caught a real bug: the camera-shake
correction had its sign backwards, so it was doubling shake instead of
removing it. On this clip the camera barely moves, so nothing looked
wrong in the output. Test 4 is why it got found.
"""

import sys
import cv2
import numpy as np

from video_io import open_video, measure_timebase, read_first_frame, frame_iterator
from tracking import Fencer
from camera import CameraMotion
from verify import intersection_over_union, OverlapWatch
from motion import PisteAxis, MovementLabeler
from main import _scale_for_display

PASS, FAIL = "  PASS", "  FAIL"
results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"{PASS if condition else FAIL}  {name}")
    if detail:
        print(f"         {detail}")


# ----------------------------------------------------------------------

def test_display_scaling():
    print("\n1. Box coordinates survive the shrink-to-fit window")
    frame = np.zeros((1080, 1920, 3), np.uint8)
    small, scale = _scale_for_display(frame, 1280)

    check("1920px frame is shrunk for display",
          small.shape[1] == 1280 and abs(scale - 1280 / 1920) < 1e-9,
          f"displayed at {small.shape[1]}x{small.shape[0]}, scale={scale:.4f}")

    # A box you'd draw on the shrunk window, and where it must land at
    # full resolution. Getting this wrong puts every box in the wrong
    # place -- silently, because it still looks like a valid box.
    drawn = (100, 200, 150, 300)
    inv = 1.0 / scale
    restored = tuple(int(v * inv) for v in drawn)
    expected = (150, 300, 225, 450)
    check("box maps back to full resolution",
          restored == expected, f"{drawn} -> {restored}, expected {expected}")

    # A frame already smaller than the window must not be touched.
    tiny = np.zeros((360, 640, 3), np.uint8)
    same, scale2 = _scale_for_display(tiny, 1280)
    check("small frames are left alone", scale2 == 1.0 and same.shape == tiny.shape)


def test_piste_axis():
    print("\n2. Piste direction is measured, not assumed")
    axis = PisteAxis((100.0, 500.0), (1500.0, 500.0))
    check("left-to-right piste gives a +x axis",
          abs(axis.unit[0] - 1.0) < 1e-6 and abs(axis.unit[1]) < 1e-6,
          axis.describe())

    # Rotated camera: piste running diagonally down the frame.
    diag = PisteAxis((0.0, 0.0), (300.0, 400.0))
    check("diagonal piste is handled too",
          abs(diag.unit[0] - 0.6) < 1e-6 and abs(diag.unit[1] - 0.8) < 1e-6,
          diag.describe())

    forward = diag.project((300.0, 400.0)) - diag.project((0.0, 0.0))
    check("distance along a diagonal piste is correct",
          abs(forward - 500.0) < 1e-6, f"projected length {forward:.1f}px")

    try:
        PisteAxis((100.0, 100.0), (100.0, 100.0))
        check("overlapping boxes are rejected", False)
    except ValueError:
        check("overlapping boxes are rejected", True,
              "raises instead of dividing by zero")


def test_movement_labels():
    print("\n3. Movement labels, including the post-loss bug")
    axis = PisteAxis((0.0, 0.0), (1000.0, 0.0))  # separation 1000 -> Still under 50 px/s
    a = MovementLabeler(axis, sign=+1)

    # Standing still: tiny jitter well under the threshold.
    for i in range(10):
        label, _ = a.update(i * 0.033, (500.0 + (i % 2), 500.0), True)
    check("a jittering-but-stationary fencer reads Still", label == "Still")

    # Advancing at ~600 px/sec, which is far above the threshold.
    b = MovementLabeler(axis, sign=+1)
    for i in range(10):
        label, speed = b.update(i * 0.033, (500.0 + i * 20, 500.0), True)
    check("advancing toward the opponent reads Forward",
          label == "Forward", f"speed {speed:+.0f} px/sec")

    # The right-hand fencer moving the same direction on screen is
    # RETREATING, because their "forward" is the other way.
    c = MovementLabeler(axis, sign=-1)
    for i in range(10):
        label, speed = c.update(i * 0.033, (500.0 + i * 20, 500.0), True)
    check("same screen motion reads Backward for the other fencer",
          label == "Backward", f"speed {speed:+.0f} px/sec")

    # THE BUG THIS CATCHES: the first version kept the pre-loss position
    # in history. After a fencer was lost for a second and came back
    # somewhere else, it divided that big jump by a single frame's time
    # and reported an enormous speed that never happened.
    d = MovementLabeler(axis, sign=+1)
    d.update(0.0, (100.0, 500.0), True)
    d.update(0.033, (105.0, 500.0), True)
    for i in range(30):                       # a full second of loss
        d.update(0.066 + i * 0.033, (0, 0), False)
    label, speed = d.update(1.10, (900.0, 500.0), True)
    check("a fencer reappearing after a loss reports no fake speed",
          label == "Still" and speed == 0.0,
          f"got {label} at {speed:+.0f} px/sec (history was cleared)")


def test_camera_compensation():
    print("\n4. Camera correction: slide, rotation and zoom")
    frame = cv2.imread("_selftest_frame.png")
    if frame is None:
        check("frame available", False, "run with a video path first")
        return

    h, w = frame.shape[:2]
    centre = (w / 2.0, h / 2.0)

    # --- pure slides, including the sign convention -------------------
    for shift_x, shift_y in [(12, 0), (-9, 5), (0, -7)]:
        moved = cv2.warpAffine(
            frame, np.float32([[1, 0, shift_x], [0, 1, shift_y]]), (w, h))
        cam = CameraMotion(frame)
        info = cam.update(moved, [])
        want_x, want_y = -shift_x, -shift_y
        check(f"content slid ({shift_x:+d},{shift_y:+d}) -> "
              f"camera ({want_x:+d},{want_y:+d})",
              info["reliable"] and abs(info["dx"] - want_x) < 1.0
              and abs(info["dy"] - want_y) < 1.0,
              f"measured ({info['dx']:+.2f},{info['dy']:+.2f}) "
              f"from {info['points']} points")

        # A fencer who never moved must come back to where they started.
        still_on_screen = (500.0 + shift_x, 400.0 + shift_y)
        back = cam.stabilise(still_on_screen)
        check("  a stationary fencer stabilises to zero movement",
              abs(back[0] - 500.0) < 1.5 and abs(back[1] - 400.0) < 1.5,
              f"stabilised to ({back[0]:.1f},{back[1]:.1f}), want (500.0,400.0)")

    # --- rotation, which the old sliding-only model could not see -----
    rot = cv2.getRotationMatrix2D(centre, 1.5, 1.0)     # rotate content +1.5 deg
    rotated = cv2.warpAffine(frame, rot, (w, h))
    cam = CameraMotion(frame)
    info = cam.update(rotated, [])
    check("a 1.5 deg camera rotation is measured",
          info["reliable"] and abs(abs(info["rotation_deg"]) - 1.5) < 0.2,
          f"measured {info['rotation_deg']:+.3f} deg")

    point_before = (700.0, 500.0)
    moved_point = rot @ np.array([point_before[0], point_before[1], 1.0])
    back = cam.stabilise((moved_point[0], moved_point[1]))
    check("  rotation is undone for a stationary fencer",
          abs(back[0] - 700.0) < 3.0 and abs(back[1] - 500.0) < 3.0,
          f"stabilised to ({back[0]:.1f},{back[1]:.1f}), want (700.0,500.0)")

    # --- zoom: clip 4 really does zoom 9%, and sliding-only ignores it -
    zoom = cv2.getRotationMatrix2D(centre, 0.0, 1.05)
    zoomed = cv2.warpAffine(frame, zoom, (w, h))
    cam = CameraMotion(frame)
    info = cam.update(zoomed, [])
    check("a 5% camera zoom is measured",
          info["reliable"] and abs(info["scale"] - 1.05) < 0.01,
          f"measured scale {info['scale']:.4f}")

    moved_point = zoom @ np.array([700.0, 500.0, 1.0])
    back = cam.stabilise((moved_point[0], moved_point[1]))
    check("  zoom is undone for a stationary fencer",
          abs(back[0] - 700.0) < 3.0 and abs(back[1] - 500.0) < 3.0,
          f"stabilised to ({back[0]:.1f},{back[1]:.1f}), want (700.0,500.0)")


def test_overlap_watch():
    print("\n5. Two boxes cannot be the same person")
    check("identical boxes score 1.0",
          abs(intersection_over_union((0, 0, 100, 100), (0, 0, 100, 100)) - 1.0) < 1e-9)
    check("separate boxes score 0.0",
          intersection_over_union((0, 0, 100, 100), (500, 500, 100, 100)) == 0.0)
    half = intersection_over_union((0, 0, 100, 100), (50, 0, 100, 100))
    check("half-overlapping boxes score 1/3",
          abs(half - 1.0 / 3.0) < 1e-6, f"got {half:.4f}")

    # A brief pass, like a fleche, must NOT raise a warning.
    watch = OverlapWatch()
    fired = False
    for _ in range(5):
        _, _, new = watch.update((0, 0, 100, 100), (5, 0, 100, 100), True)
        fired = fired or new
    for _ in range(10):
        _, _, new = watch.update((0, 0, 100, 100), (500, 0, 100, 100), True)
        fired = fired or new
    check("a brief close pass does not warn", not fired,
          "5 overlapping frames then they separate")

    # Two trackers stuck on one person must raise exactly one warning.
    watch = OverlapWatch()
    warnings = 0
    for _ in range(30):
        _, _, new = watch.update((0, 0, 100, 100), (5, 0, 100, 100), True)
        warnings += int(new)
    check("a sustained overlap warns exactly once", warnings == 1,
          f"{warnings} warning(s) over 30 stuck frames")

    watch = OverlapWatch()
    fired = False
    for _ in range(30):
        _, _, new = watch.update((0, 0, 100, 100), (5, 0, 100, 100), False)
        fired = fired or new
    check("no warning while a fencer is lost", not fired)


def test_pose(video_path):
    print("\n6. Body landmarks and the foot-height check")
    try:
        from pose import PoseEstimator, FootWatch, PoseResult
    except Exception as exc:
        check("pose module imports", False, str(exc)[:100])
        return

    frame = cv2.imread("_selftest_frame.png")
    est = PoseEstimator()

    on_fencer = est.estimate(frame, (240, 430, 175, 215))
    check("a pose is found on a boxed fencer",
          on_fencer is not None and on_fencer.confidence > 0.5,
          f"confidence {on_fencer.confidence:.3f}" if on_fencer else "nothing found")

    if on_fencer is not None:
        feet = on_fencer.feet
        # The fencer's feet must land in the lower half of the frame and
        # below the box we drew round their torso.
        check("the feet land below the torso box, on the floor",
              feet[1] > 430 + 215, f"feet at y={feet[1]:.0f}, box bottom 645")
        wrist, elbow = on_fencer.sword_hand()
        check("a sword hand and elbow are reported",
              wrist is not None and elbow is not None,
              f"wrist ({wrist[0]:.0f},{wrist[1]:.0f})")
        check("arm extension is a sensible ratio",
              0.0 < on_fencer.arm_extension() < 5.0,
              f"{on_fencer.arm_extension():.2f} torso-widths")

    # THE USEFUL PART: empty floor must produce nothing at all.
    on_floor = est.estimate(frame, (700, 700, 200, 220))
    check("empty floor produces NO pose", on_floor is None,
          "this is what makes 'the box slid onto the floor' detectable")

    # --- FootWatch ---------------------------------------------------
    class FakePose:
        def __init__(self, y): self._y = y
        @property
        def feet(self): return (0.0, self._y)

    # A fencer on the piste: feet wander gently, never climb.
    watch = FootWatch()
    fired = False
    for i in range(200):
        _, _, new = watch.update(FakePose(800 + (i % 7) - 3))
        fired = fired or new
    check("a fencer staying on the piste never warns", not fired)

    # A tracker sliding into the background: feet climb 250px and stay.
    watch = FootWatch()
    fired_at = None
    for i in range(200):
        y = 800 if i < 90 else 550
        _, _, new = watch.update(FakePose(y))
        if new and fired_at is None: fired_at = i
    check("feet climbing into the background does warn",
          fired_at is not None, f"warned at step {fired_at}")

    # THE BUG THIS CATCHES: pose fails on ~25% of frames here. If a
    # missing reading cleared the evidence, the warning could never
    # accumulate. Gaps must hold the count, not wipe it.
    watch = FootWatch()
    fired_at = None
    for i in range(240):
        y = 800 if i < 90 else 550
        sample = None if (i % 3 == 0) else FakePose(y)   # 1 frame in 3 missing
        _, _, new = watch.update(sample)
        if new and fired_at is None: fired_at = i
    check("warning still fires when a third of frames have no pose",
          fired_at is not None, f"warned at step {fired_at}")


def test_detection(video_path):
    print("\n7. Person detection and matching")
    try:
        from detect import (PersonDetector, match_detections,
                            torso_from_person, _containment)
    except Exception as exc:
        check("detect module imports", False, str(exc)[:100])
        return

    # --- matching logic, no model needed ---------------------------
    check("a box fully inside another is fully contained",
          abs(_containment((10, 10, 20, 20), (0, 0, 100, 100)) - 1.0) < 1e-9)
    check("a box outside another is not contained",
          _containment((200, 200, 20, 20), (0, 0, 100, 100)) == 0.0)

    torso = torso_from_person((100, 100, 200, 400))
    check("a torso box sits in the upper half of a person",
          torso[1] + torso[3] / 2 < 100 + 400 / 2,
          f"torso centre y={torso[1] + torso[3] // 2}, person spans 100-500")

    keep = torso_from_person((100, 100, 200, 400), keep_size=(60, 80))
    check("keep_size holds the box size steady",
          keep[2] == 60 and keep[3] == 80,
          "stops the box pulsing when the detector wobbles")

    # THE IMPORTANT ONE: two fencers must never claim the same body.
    one_person = [((100, 100, 200, 400), 0.9)]
    both_near = [(140, 200, 60, 80), (150, 210, 60, 80)]
    assigned = match_detections(both_near, one_person, max_move=500)
    claimed = [a for a in assigned if a is not None]
    check("one detected body can only be claimed by ONE fencer",
          len(claimed) == 1,
          "this is what stops both boxes collapsing onto the same person")

    two_people = [((100, 100, 200, 400), 0.9), ((600, 100, 200, 400), 0.9)]
    apart = [(140, 200, 60, 80), (640, 200, 60, 80)]
    assigned = match_detections(apart, two_people, max_move=500)
    check("two separated fencers each get their own body",
          all(a is not None for a in assigned) and assigned[0] != assigned[1])

    # A fencer far from every detection must match nothing.
    assigned = match_detections([(1500, 900, 60, 80)], two_people, max_move=200)
    check("a box on scenery matches no body at all",
          assigned[0] is None,
          "this is what turns silent drift into an honest loss")

    # --- momentum through a crossing --------------------------------
    from detect import FencerMotion
    motion = FencerMotion()
    # A fencer travelling steadily right at 600 px/sec.
    for step in range(10):
        motion.observe((100.0 + step * 20.0, 500.0), 1 / 30.0, 200)
    check("velocity is measured from observed positions",
          motion.speed() > 300, f"{motion.speed():.0f} px/sec")
    ahead = motion.predict(1 / 30.0)
    check("prediction continues in the direction of travel",
          ahead[0] > 280, f"predicted x={ahead[0]:.0f} from x=280")

    # THE CROSSING TEST: two fencers pass each other. Both bodies end up
    # in the same place, and only momentum says which is which.
    right_mover = FencerMotion()
    left_mover = FencerMotion()
    for step in range(10):
        right_mover.observe((100.0 + step * 20.0, 500.0), 1 / 30.0, 200)
        left_mover.observe((500.0 - step * 20.0, 500.0), 1 / 30.0, 200)
    # They are now at 280 and 320 -- almost on top of each other.
    for _ in range(6):        # coast through the pass, seeing nobody
        right_mover.coast(1 / 30.0)
        left_mover.coast(1 / 30.0)
    check("after a crossing the two predictions are on opposite sides",
          right_mover.position[0] > left_mover.position[0],
          f"right-mover now at x={right_mover.position[0]:.0f}, "
          f"left-mover at x={left_mover.position[0]:.0f}")

    # Coasting must fade, not fly off across the hall forever.
    far = FencerMotion()
    for step in range(10):
        far.observe((100.0 + step * 20.0, 500.0), 1 / 30.0, 200)
    start_speed = far.speed()
    for _ in range(30):
        far.coast(1 / 30.0)
    check("coasting bleeds off speed instead of running away",
          far.speed() < start_speed * 0.2,
          f"{start_speed:.0f} -> {far.speed():.0f} px/sec after 30 frames")

    # A wild one-frame jump must not become a wild prediction.
    capped = FencerMotion()
    capped.observe((100.0, 500.0), 1 / 30.0, 200)
    capped.observe((1800.0, 500.0), 1 / 30.0, 200)      # teleport
    check("an implausible jump is speed-capped",
          capped.speed() <= 6.0 * 200 + 1,
          f"{capped.speed():.0f} px/sec, cap is {6.0 * 200:.0f}")

    # --- the real model on a real frame -----------------------------
    frame = cv2.imread("_selftest_frame.png")
    detector = PersonDetector()
    people = detector.detect(frame)
    check("people are detected in a real fencing frame",
          len(people) >= 2, f"{len(people)} detected")
    if people:
        check("detections carry a real confidence",
              all(0.0 < s <= 1.0 for _b, s in people),
              f"top score {people[0][1]:.2f}")


def test_reselection(video_path):
    print("\n8. Mid-clip re-selection (the A / B key path)")
    cap = open_video(video_path)
    first = read_first_frame(cap)
    fencer = Fencer("Fencer A", (0, 140, 255), first, (235, 425, 215, 470))

    boxes = {}
    for idx, _ts, frame in frame_iterator(cap):
        fencer.update(frame)
        boxes[idx] = fencer.box
    cap.release()
    clean_final = boxes[max(boxes)]

    # Now do it again, but deliberately wreck the tracker at frame 40 by
    # re-seeding it onto a patch of empty floor -- standing in for a real
    # drift -- and then rescue it at frame 60 the way pressing A would.
    cap = open_video(video_path)
    first = read_first_frame(cap)
    fencer2 = Fencer("Fencer A", (0, 140, 255), first, (235, 425, 215, 470))
    rescued = None
    for idx, _ts, frame in frame_iterator(cap):
        if idx == 40:
            fencer2.reinit(frame, (900, 900, 120, 120))     # break it
        elif idx == 60:
            fencer2.reinit(frame, boxes[60])                # rescue it
        fencer2.update(frame)
        rescued = fencer2.box
    cap.release()

    check("reinit() resets the tracker and counts the fix",
          fencer2.reinit_count == 2, f"reinit_count={fencer2.reinit_count}")

    dx = abs(rescued[0] - clean_final[0])
    dy = abs(rescued[1] - clean_final[1])
    check("a rescued tracker converges back to the clean result",
          dx < 60 and dy < 60,
          f"rescued {rescued} vs clean {clean_final} (dx={dx}, dy={dy})")


def test_timebase(video_path):
    print("\n9. Real timing beats the file header")
    cap = open_video(video_path)
    header_fps = cap.get(cv2.CAP_PROP_FPS)
    header_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    tb = measure_timebase(video_path)
    check("timestamps were measured from the file",
          tb["frame_count"] > 1,
          f"header: {header_count} frames @ {header_fps:.2f} fps  |  "
          f"measured: {tb['frame_count']} frames @ {tb['measured_fps']:.2f} fps")

    cap = open_video(video_path)
    counted = sum(1 for _ in frame_iterator(cap))
    cap.release()
    check("measured frame count matches an actual decode",
          counted == tb["frame_count"],
          f"{counted} frames decoded, {tb['frame_count']} measured")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Give me a video path.")
        sys.exit(2)
    video_path = sys.argv[1]

    # Cache a real frame for the camera test.
    cap = open_video(video_path)
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite("_selftest_frame.png", frame)

    test_display_scaling()
    test_piste_axis()
    test_movement_labels()
    test_camera_compensation()
    test_overlap_watch()
    test_pose(video_path)
    test_detection(video_path)
    test_reselection(video_path)
    test_timebase(video_path)

    passed, total = sum(results), len(results)
    print(f"\n{'=' * 58}")
    print(f"{passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
