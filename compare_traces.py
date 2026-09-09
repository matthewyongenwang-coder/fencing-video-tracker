"""
compare_traces.py

Diffs two run traces and tells me whether the tracker's behaviour moved.

    python3 compare_traces.py golden.json new.json

Exit code 0 means nothing moved. Anything else means it did, and the output
says where.

WHAT COUNTS AS A DIFFERENCE
Not everything in a trace deserves the same tolerance, and pretending it
does is how you end up either ignoring real breakage or chasing float noise
for a week.

Discrete fields are compared exactly, with no tolerance at all. Whether a
fencer is tracked, which movement label was reported, whether the overlap or
feet warning fired, how many corner points the camera estimate used, and
every LOST or SNAP event with its frame number. These are the things the
program actually reasons about, and the things I read when I am deciding
whether a run went well. If one of them moves, behaviour changed.

Continuous fields are positions and speeds. By default these are compared
exactly too, because I measured the Python engine against itself on the
fleche clip and the CSVs came back byte-identical. When the reference is
deterministic there is no reason to accept drift from it, and accepting it
anyway means a real regression can hide under the allowance.

--budget switches to the tolerances a port is allowed: box centres within
2px, stabilised positions within 3px, speeds within 5px/s or 2%. That mode
exists for one job, moving this engine to another language, where the
feedback loops inside CSRT and the chained camera transform make exact
agreement genuinely impossible. It is not for excusing a change to the
Python engine, so it stays off unless you ask for it.

WHAT THIS DOES NOT TELL YOU
That either run was right. Both traces can agree perfectly while both boxes
sit on a referee. That is what EXPECTED in benchmark.py and my own eyes on
the filmstrip are for. Agreement plus an unchanged EXPECTED verdict is the
pair that means something.

WHY IT REPORTS THE FIRST DIVERGENT FRAME
A total tells me a run is broken and nothing else. The frame where two runs
first stop agreeing is usually a straight line to the cause, and everything
after it is downstream of that one moment.
"""

import argparse
import os
import sys

import runtrace

# What a port is allowed to differ by. Unused unless --budget is passed.
BUDGET_CENTER_PX = 2.0
BUDGET_STABLE_PX = 3.0
BUDGET_SPEED_PX_S = 5.0
BUDGET_SPEED_REL = 0.02

# Header fields that must match before a comparison means anything. A run on
# a different clip, or with a different model file, is not a regression.
MUST_MATCH_HEADER = ("video_name", "start_frame", "max_frames", "seed_box_a",
                     "seed_box_b", "detect", "pose", "tracker", "models",
                     "width", "height", "frame_count", "trace_format")


class Report:
    def __init__(self):
        self.setup_problems = []   # ran them differently, not a regression
        self.discrete = []         # behaviour changed
        self.continuous = []       # numbers moved beyond budget
        self.event_problems = []
        self.first_divergent = None
        self.worst = {"center": 0.0, "stable": 0.0, "speed": 0.0}

    def note_frame(self, frame_index):
        # The earliest one, not the first one reported. Events are checked
        # before frames and in sorted order, so keeping whichever arrived
        # first pointed at frame 195 when the real divergence was at 187.
        # This number is the one I act on, so it has to be the true
        # minimum.
        if self.first_divergent is None or frame_index < self.first_divergent:
            self.first_divergent = frame_index

    @property
    def ok(self):
        return not (self.setup_problems or self.discrete or
                    self.continuous or self.event_problems)


def _compare_header(a, b, report):
    for key in MUST_MATCH_HEADER:
        left, right = a["header"].get(key), b["header"].get(key)
        if left != right:
            report.setup_problems.append(f"header {key}: {left!r} vs {right!r}")
    # Version drift is worth saying out loud but is not itself a failure,
    # because it is the explanation for a difference rather than the
    # difference.
    for key in ("opencv_version", "python_version", "platform"):
        left, right = a["header"].get(key), b["header"].get(key)
        if left != right:
            print(f"  note: {key} differs, {left} vs {right}")


def _compare_events(a, b, report):
    def key(event):
        return (event["frame"], event["kind"], event["who"])

    left = sorted(key(e) for e in a["events"])
    right = sorted(key(e) for e in b["events"])
    for missing in sorted(set(left) - set(right)):
        report.event_problems.append(
            f"event vanished: {missing[1]} for {missing[2]} at frame {missing[0]}")
        report.note_frame(missing[0])
    for added in sorted(set(right) - set(left)):
        report.event_problems.append(
            f"event appeared: {added[1]} for {added[2]} at frame {added[0]}")
        report.note_frame(added[0])


def _num_diff(left, right):
    """Absolute difference, treating a missing value as a mismatch."""
    if left is None and right is None:
        return None
    if left is None or right is None:
        return float("inf")
    return abs(float(left) - float(right))


def _compare_frames(a, b, report, budget):
    left_frames, right_frames = a["frames"], b["frames"]
    if len(left_frames) != len(right_frames):
        report.discrete.append(
            f"frame count: {len(left_frames)} vs {len(right_frames)}")

    for left, right in zip(left_frames, right_frames):
        index = left["frame"]
        if right["frame"] != index:
            report.discrete.append(
                f"frame index out of step: {index} vs {right['frame']}")
            report.note_frame(index)
            break

        # --- discrete, always exact ---------------------------------
        for field in ("reliable", "points"):
            if left["camera"][field] != right["camera"][field]:
                report.discrete.append(
                    f"frame {index}: camera {field} "
                    f"{left['camera'][field]} vs {right['camera'][field]}")
                report.note_frame(index)
        if left["overlap"]["warning"] != right["overlap"]["warning"]:
            report.discrete.append(f"frame {index}: overlap warning changed")
            report.note_frame(index)

        for side, (lf, rf) in enumerate(zip(left["fencers"], right["fencers"])):
            who = "A" if side == 0 else "B"
            for field in ("tracked", "movement", "detected", "feet_warning"):
                if lf[field] != rf[field]:
                    report.discrete.append(
                        f"frame {index}: fencer {who} {field} "
                        f"{lf[field]!r} vs {rf[field]!r}")
                    report.note_frame(index)

            # A pose found in one run and not the other is a behaviour
            # change, not a rounding difference.
            if (lf["pose"] is None) != (rf["pose"] is None):
                report.discrete.append(
                    f"frame {index}: fencer {who} pose found in one run only")
                report.note_frame(index)

            # --- continuous ------------------------------------------
            if not lf["tracked"] or not rf["tracked"]:
                continue
            for field, limit, bucket in (
                    ("center", BUDGET_CENTER_PX, "center"),
                    ("stable", BUDGET_STABLE_PX, "stable")):
                for axis in (0, 1):
                    gap = _num_diff(lf[field][axis], rf[field][axis])
                    if gap is None:
                        continue
                    report.worst[bucket] = max(report.worst[bucket], gap)
                    allowed = limit if budget else 0.0
                    if gap > allowed:
                        report.continuous.append(
                            f"frame {index}: fencer {who} {field}"
                            f"{'xy'[axis]} off by {gap:.2f}px")
                        report.note_frame(index)

            gap = _num_diff(lf["speed_px_s"], rf["speed_px_s"])
            if gap is not None:
                report.worst["speed"] = max(report.worst["speed"], gap)
                if budget:
                    scale = max(abs(lf["speed_px_s"] or 0.0), 1.0)
                    allowed = max(BUDGET_SPEED_PX_S, BUDGET_SPEED_REL * scale)
                else:
                    allowed = 0.0
                if gap > allowed:
                    report.continuous.append(
                        f"frame {index}: fencer {who} speed off by "
                        f"{gap:.1f}px/s")
                    report.note_frame(index)


def compare(golden_path, new_path, budget=False):
    a = runtrace.load(golden_path)
    b = runtrace.load(new_path)
    report = Report()
    _compare_header(a, b, report)
    if report.setup_problems:
        # No point diffing the frames of two different runs.
        return report
    _compare_events(a, b, report)
    _compare_frames(a, b, report, budget)
    return report


def _print(report, golden_path, new_path, budget, limit):
    print(f"golden : {golden_path}")
    print(f"new    : {new_path}")
    print(f"mode   : {'budget (port tolerances)' if budget else 'exact'}")

    if report.setup_problems:
        print("\nThese two runs are not comparable:")
        for line in report.setup_problems:
            print(f"  {line}")
        print("\nThat is a difference in how they were run, not a regression. "
              "Fix the command and run again.")
        return

    print("\nlargest movement seen, whether or not it is allowed:")
    print(f"  box centre        {report.worst['center']:.2f} px")
    print(f"  stabilised centre {report.worst['stable']:.2f} px")
    print(f"  speed             {report.worst['speed']:.1f} px/s")

    for title, lines in (("Events", report.event_problems),
                         ("Behaviour", report.discrete),
                         ("Numbers", report.continuous)):
        if not lines:
            continue
        print(f"\n{title}, {len(lines)} problem(s):")
        for line in lines[:limit]:
            print(f"  {line}")
        if len(lines) > limit:
            print(f"  ... and {len(lines) - limit} more")

    if report.first_divergent is not None:
        print(f"\nFirst frame where the two runs stop agreeing: "
              f"{report.first_divergent}")

    print()
    print("MATCH" if report.ok else "CHANGED")


def main():
    parser = argparse.ArgumentParser(description="Diff two tracker run traces.")
    parser.add_argument("golden", help="the trace to measure against")
    parser.add_argument("new", help="the trace to check")
    parser.add_argument("--budget", action="store_true",
                        help="allow the tolerances a port to another language "
                             "is permitted. Off by default, because the "
                             "Python engine is deterministic and should "
                             "match itself exactly.")
    parser.add_argument("--limit", type=int, default=15,
                        help="how many problems of each kind to print")
    args = parser.parse_args()

    for path in (args.golden, args.new):
        if not os.path.exists(path):
            # Say which file, rather than a traceback. A missing trace
            # nearly always means the run that should have produced it
            # failed, and that is the thing worth going and looking at.
            print(f"No trace at {path}. The run that writes it probably "
                  f"did not finish.")
            return 2

    report = compare(args.golden, args.new, budget=args.budget)
    _print(report, args.golden, args.new, args.budget, args.limit)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
