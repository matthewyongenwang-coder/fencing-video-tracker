"""
golden.py

Records what the tracker currently does on every benchmark case, and later
checks that it still does it.

    python3 golden.py record     # save today's behaviour as the reference
    python3 golden.py check      # has anything moved since?
    python3 golden.py check sf seattle    # just those cases

check exits non-zero when behaviour changed, so it is the one command to
run before committing anything that touches the engine.

WHY THIS IS SEPARATE FROM benchmark.py
benchmark.py answers "is the box on the right person", which only I can
answer, by looking at a filmstrip. This answers "did anything change since
last time", which a computer can answer and I cannot, because the change
might be one pixel on frame 143 of 221.

The two are not interchangeable and neither replaces the other. A run can
pass check while tracking a referee. A run can fail check because I
deliberately improved something. Read together they tell you what you need:
check says what moved, and the filmstrip says whether the move was an
improvement.

WHY IT SHELLS OUT TO main.py
Because main.py is the real program. benchmark.py has its own copy of the
tracking loop, which means there are already two implementations in this
repo that can drift apart, and I am not going to add a third to test the
first two. Recording through main.py means a golden trace is a record of
what actually runs when I use this thing.

WHAT IS NOT COVERED
Video decoding. These runs read the original .MOV files, so a trace proves
the engine is stable on this machine with this OpenCV. It does not prove
anything about a different decoder, because FFmpeg, AVFoundation and
MediaCodec do not hand back identical pixels from the same file. When the
engine moves off this machine, the cases need to run from extracted frames
instead, or every decoder difference will look like an algorithm bug.
"""

import argparse
import os
import subprocess
import sys

import compare_traces
from benchmark import CASES

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, "golden")

# Recorded with detection on. That is the configuration I actually use on
# real footage, and it is the one the README's measurements are about. Pose
# is on too, because the foot-height warning is part of the behaviour worth
# protecting and it does not exist without it.
RUN_ARGS = ["--detect", "--pose", "--no-display"]


def _box(values):
    return ",".join(str(int(v)) for v in values)


def pick(names):
    if not names:
        return list(CASES)
    chosen = []
    for case in CASES:
        if any(name.lower() in case[0].lower() for name in names):
            chosen.append(case)
    return chosen


def run_one(case, trace_path, work_dir):
    """Run main.py over one case and leave a trace at trace_path."""
    name, video, start, count, box_a, box_b = case
    if not os.path.exists(video):
        return None, f"clip missing: {video}"

    command = [
        sys.executable, os.path.join(HERE, "main.py"), video,
        "--box-a", _box(box_a),
        "--box-b", _box(box_b),
        "--start-frame", str(start),
        "--max-frames", str(count),
        "--out-dir", os.path.join(work_dir, name),
        "--trace", trace_path,
    ] + RUN_ARGS

    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0:
        tail = (finished.stderr or finished.stdout).strip().splitlines()
        return None, "run failed: " + (tail[-1] if tail else "no output")
    if not os.path.exists(trace_path):
        return None, "run finished but wrote no trace"
    return trace_path, None


def record(names):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    cases = pick(names)
    print(f"Recording {len(cases)} case(s) into {GOLDEN_DIR}\n")
    failures = 0
    for case in cases:
        name = case[0]
        print(f"  {name:<18}", end="", flush=True)
        path = os.path.join(GOLDEN_DIR, f"{name}.json")
        _, problem = run_one(case, path, os.path.join(HERE, "_golden_work"))
        if problem:
            print(f"SKIPPED, {problem}")
            failures += 1
            continue
        trace = compare_traces.runtrace.load(path)
        losses = [e for e in trace["events"] if e["kind"] == "LOST"]
        print(f"recorded, {len(trace['frames'])} frames, "
              f"{len(losses)} loss event(s)")
    print()
    if failures:
        print(f"{failures} case(s) could not be recorded. The traces that did "
              f"get written are still usable.")
    return 1 if failures else 0


def check(names, budget=False):
    cases = pick(names)
    print(f"Checking {len(cases)} case(s) against {GOLDEN_DIR}\n")
    changed, skipped = [], []
    for case in cases:
        name = case[0]
        golden_path = os.path.join(GOLDEN_DIR, f"{name}.json")
        print(f"  {name:<18}", end="", flush=True)
        if not os.path.exists(golden_path):
            print("no golden trace, run `golden.py record` first")
            skipped.append(name)
            continue

        fresh_path = os.path.join(HERE, "_golden_work", f"{name}.check.json")
        _, problem = run_one(case, fresh_path,
                             os.path.join(HERE, "_golden_work"))
        if problem:
            print(f"SKIPPED, {problem}")
            skipped.append(name)
            continue

        report = compare_traces.compare(golden_path, fresh_path, budget=budget)
        if report.ok:
            print("match")
        else:
            changed.append((name, report))
            print(f"CHANGED, {len(report.event_problems)} event, "
                  f"{len(report.discrete)} behaviour, "
                  f"{len(report.continuous)} number problem(s), first at "
                  f"frame {report.first_divergent}")

    print()
    for name, report in changed:
        print(f"--- {name} ---")
        for line in (report.setup_problems + report.event_problems +
                     report.discrete + report.continuous)[:10]:
            print(f"  {line}")
        print()

    if changed:
        print(f"{len(changed)} case(s) changed. If that was on purpose, look "
              f"at the filmstrips from benchmark.py to confirm it is an "
              f"improvement, write the measured size into the README, then "
              f"re-record.")
        return 1
    if skipped:
        print(f"Nothing moved, but {len(skipped)} case(s) were skipped: "
              f"{', '.join(skipped)}")
        return 0
    print("Nothing moved.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Record and check golden traces for the benchmark cases.")
    parser.add_argument("action", choices=["record", "check"])
    parser.add_argument("cases", nargs="*",
                        help="case names, or none for all of them")
    parser.add_argument("--budget", action="store_true",
                        help="allow the tolerances a port to another language "
                             "is permitted. See compare_traces.py.")
    args = parser.parse_args()

    if args.action == "record":
        return record(args.cases)
    return check(args.cases, budget=args.budget)


if __name__ == "__main__":
    sys.exit(main())
