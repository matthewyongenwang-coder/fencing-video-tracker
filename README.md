# Riposte

Track two fencers through competition video. You pick them once, and the
program follows them, writes out an annotated clip and a CSV of positions
and speeds, and tells you when it has lost them.

![Two fencers tracked with boxes and body landmarks](docs/demo.gif)

I fence sabre and I film a lot of my bouts. I wanted something that could
follow both fencers automatically so I could pull numbers out of the
footage instead of scrubbing through it by hand.

Python and OpenCV, two pip packages, no cloud and no accounts. It runs
from the terminal or from a local web page.

Before anything else, read [FILMING.md](FILMING.md). How you film changes
the results more than anything in this code does.

---

## What it does

- Follows two fencers you select on the first frame, keeping them labelled
  A and B.
- Uses a person detector so a box can only sit on an actual person.
- Carries identities through a fleche crossing using momentum.
- Removes camera slide, rotation and zoom before measuring any movement.
- Finds body landmarks: feet, wrists, joints.
- Labels movement along the piste as Forward, Backward or Still.
- Says LOST when it cannot find the fencer, instead of pretending.

That last one took the most work and is the part I care about most. A
tracker that quietly follows the wrong person is worse than one that
stops.

---

## Setup

```bash
git clone https://github.com/matthewyongenwang-coder/riposte.git
cd riposte
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 get_models.py
```

Two pip packages. `get_models.py` fetches about 42MB of pretrained models
from the OpenCV Model Zoo. They are not in the repo because they never
change and would bloat every clone. I did not train any of them.

Plain tracking works with no models at all. You need them for `--detect`,
which I recommend on real footage, and for `--pose`.

## Two ways to run it

A web page:

```bash
python3 webapp.py
```

Then open http://127.0.0.1:8765. Drop in a clip, scrub to the action, drag
a box around each fencer, press Run.

Or the terminal:

```bash
python3 main.py "~/Videos/Fencing/bout.MOV" --detect --pose
```

Keep the quotes if the path has spaces. Inside Python a path with spaces
is just a string, so quoting is only a shell requirement.

While it runs, SPACE pauses, A and B re-select a fencer, and Q or ESC
stops early and still saves everything processed so far.

Useful flags:

```bash
--start-frame 1950    # clips are whole bouts, skip to the phrase
--max-frames 150      # how much to process
--detect              # recommended, see below
--pose                # body landmarks
--no-display          # batch, no window
```

Check nothing is broken:

```bash
python3 selftest.py "/path/to/a/clip.MOV"
```

53 checks, all passing.

---

## Draw the box around the mask and torso

This is the single biggest thing you control, and it matters more than any
setting in the code. On the `portland-fleche` test clip:

| What I boxed | What happened |
|---|---|
| Whole fencer, 270x450 | Slid onto the referee at frame 49 and tracked him for the next 170 frames, reporting success the whole time |
| Mask and torso, 205x240 | Followed the fencer correctly through the lunge and the retreat |

A full-body box is mostly empty floor, especially the gap between the legs
in a lunge, and legs swing around. The tracker ends up learning "pale
wooden floor", which matches floor everywhere. The mask and torso are
compact and keep their shape.

The program warns you if the box you drew looks like a full body.

---

## Where it goes wrong

I tested five segments from four competitions. Of 10 fencer-tracks, only 5
were still on the right person at the end. Every failure landed on the same
kind of thing.

| Case | What the box ended up on |
|---|---|
| portland-fleche A | a referee standing still |
| seattle-lowres B | a spectator's dark hair |
| sf-blur-posters A | the ceiling edge |
| sf-blur-posters B | the scoreboard |
| cincy-pan-blur A | a sponsor banner |

Banners, scoreboards and ceiling edges are static and high-contrast, and
CSRT prefers them to a blurred fencer mid-lunge. Template tracking has no
idea what a person is, so nothing stops it. All five reported
`tracked=True` the whole time.

The other failure is the crossing. When two fencers run past each other in
a fleche, both boxes sit over the same patch of image for several frames,
and a template tracker cannot tell which body is which coming out. Running
the same clip with a one-frame offset sent the tracker after the other
fencer, so it was close to a coin flip.

---

## What I tried that did not work

I am listing these because they explain why there is one automatic check in
the code instead of five.

| Idea | Why it failed |
|---|---|
| Cap how far a box may jump per frame | A correct box during a fast lunge jumped 0.284 times its own height. The drifting box jumped 0.356. Too close to separate. The other drift jumped only 0.194, which is less than the correct lunge. |
| Compare tracker motion against optical flow of the box contents | Correct tracking during a blurry lunge disagreed by 116px. An actual drift disagreed by 157px. Overlapping again. |
| Raise CSRT's own `psr_threshold` so it gives up sooner | At 0.06 nothing changed. At 0.10 it declared failure at frame 30, in the middle of a good lunge. |
| Other CSRT tuning: `filter_lr`, `padding`, `template_size`, `use_segmentation` | I swept all of them. Every variant still ended up on the referee. |
| Pose confidence as a drift signal | It stayed above 0.97 the whole time a box was tracking the wrong people, because those people are real bodies. |

Tuning does not fix this. The box you draw does, and so does detection.

---

## Detection is what actually fixed it

```bash
python3 main.py "clip.MOV" --detect
```

YOLOX finds every person in each frame. A fencer's box has to contain a
detected body. If it does not, for 20 frames straight, that fencer is
declared LOST instead of left sitting on scenery.

This removes the failure rather than trying to spot it afterwards. A box
cannot drift onto a banner because a banner is not a person.

It finds the fencers even when they are badly smeared: 0.84 confidence on
the heavily blurred fencer in the San Francisco clip, 0.68 on the
motion-streaked one in Cincinnati. It also does not report the large
photographs of fencers on the San Francisco back wall as people, which is
what I most expected to go wrong.

Four of the five failures are now flagged instead of silent.

Matching also enforces mutual exclusion, so one detected body can be
claimed by at most one fencer. That prevents both boxes collapsing onto the
same person.

I had to raise the patience from 12 frames to 20. Detector recall on a clip
that tracks perfectly is about 94%, and the misses are the deep lunge
frames, where a fully extended fencer stops looking like the upright person
the detector expects. Twelve frames of patience produced a false LOST on an
easy clip.

Speed, measured on my Mac: 37 fps at 416x416 input, 18 fps at 640x640. I
use 640 because it lifted recall on the fleche clip from 73% to 80%. I
tried 800 and it gained almost nothing, 81%, so it is not worth the time.

---

## Crossings, solved with momentum

Two fencers in identical kit passing each other look the same, so
appearance cannot separate them. A fleche is not ambiguous physically,
though. Both fencers are carrying momentum in opposite directions and
neither reverses instantly mid-pass, so the body still travelling right is
still Fencer A.

Each fencer now carries a velocity, measured in the camera-stabilised frame
so a pan cannot be mistaken for sprinting. Detections are matched against a
predicted position as well as the last known box, and when nobody matches,
the tracker coasts on momentum for up to half a second before giving up.

Measured on the fleche clip:

| | Before | After |
|---|---|---|
| Peak overlap between the two boxes | collapsed onto one person | 0.24, never merged |
| Fencer A at the end | silently tracking a referee at x=1275 | declared LOST |
| Identity through the pass | close to a coin flip | both boxes on distinct, correct fencers at frames 105, 140 and 165, checked by eye |

Matching against the prediction alone made things worse. On a clip that
previously tracked perfectly, box jitter produced noisy velocity, the
prediction overshot, matching failed, and the fencer coasted further off.
That feedback loop cost 24 frames on an easy clip. Matching against both
the current box and the prediction, then taking whichever fits better,
fixed it. The current box anchors ordinary frames and the prediction only
earns its keep during a crossing.

Three things are still unsolved: a box drifting onto a different person who
happens to be where the fencer was predicted, telling my fencers apart from
every other fencer in the hall, and a fencer who leaves the frame entirely.
The last one is now reported honestly as lost.

---

## Camera movement

The correction measures slide, rotation and zoom together. I started with
sliding only, which was fine for the first clip and useless for the second.

| Clip | Rotation | Zoom | Drift |
|---|---|---|---|
| Portland, steady | 0.05° | 0.9994x | 7 px |
| Portland, panning | 1.01° | 1.0924x | 164 px |

A 9.2% zoom is not a rounding error, and sliding-only correction ignored it
completely. The steady clip barely moves at all, because phone
stabilisation had already done the work, so the blur during fast actions is
the fencers moving rather than the camera.

Perspective and parallax are still not modelled. The floor stretches away
from the camera, so when the camera moves sideways, near things shift more
than far things, and one transform cannot be right for both depths. The
running transform also accumulates a little error over a long clip.

---

## Body landmarks

```bash
python3 main.py "clip.MOV" --pose
```

BlazePose draws 33 body points per fencer: shoulders, elbows, wrists, hips,
knees, ankles. It runs through cv2.dnn, so it adds no new pip dependency.

I point it at each fencer's tracked box rather than the whole frame, so the
skeleton always belongs to the fencer I selected instead of whichever body
a detector ranked first in a hall full of identical white kit. It also
means I only need to box the torso, because the model still finds the legs
and feet below it.

The ankles are what made the foot-height check possible. A fencer on a
piste has feet that move smoothly and stay low in frame, and a tracker that
has wandered into the background lands on feet that sit much higher.

| Frames | Fencer A feet y | Fencer B feet y |
|---|---|---|
| 0 to 24 | about 795, correct | about 725, correct |
| 108 to 216 | 500 to 600, background | about 660, still correct |

A rose about 290px and stayed there. B moved 65px across the whole clip.
The check fires on A at frames 125 and 158 and stays silent on B and on
both fencers of the clip that tracks correctly.

Two bugs here are worth recording. The first version compared feet against
a fast-adapting running average, which followed the drift down, so the gap
always looked small and it never fired. A reference you are measuring
movement away from must not chase that movement. It also cleared its
evidence every frame the pose model found nothing, and pose only lands on
about 75% of frames because it fails on the blurriest ones, so the gaps
wiped the count before it could trigger.

Pose does not drive the tracker. I tried letting it re-lock the box each
frame, and on the fleche clip it took Fencer A from completely lost, ending
at x=1179, to correct, ending at x=205. The same change made Fencer B
worse. A conservative version that only intervened on disagreement rejected
180 of 221 frames for B as implausible, which stripped CSRT of its
stability without giving anything back. It is a real lead and not a
finished feature, so pose measures and warns rather than steering.

---

## Blades

I wanted this to see blade actions, so I looked at the actual pixels before
deciding.

When a fencer is still, the blade is visible as a dark line about 2 to 3
pixels wide, with the guard clearly readable. During the actions I care
about it is not there at all. At frame 64 of the steady clip the blade has
smeared into the background and only the guard survives as a grey blur. At
frame 88, mid-lunge, the whole arm is a streak.

That is the footage, not the algorithm. 30fps with a phone's automatic
shutter cannot freeze a sabre blade, and no detector can find something the
sensor did not record. Filming in slow motion at 120 or 240fps is the fix,
and it is the highest-value change available.

What I can measure today without the blade is the sword hand and arm
extension, from the pose landmarks. Those are in the CSV as
`sword_wrist_x/y`, `sword_elbow_x/y` and `arm_extension`, which is reach
from shoulders to sword hand in torso-widths so it does not change just
because the fencer moved further from the camera.

Arm extension tells you the arm extended. It does not tell you there was an
attack, a parry or a hit, and in sabre right of way turns on exactly those.
The sword-hand guess is geometry too. It picks the wrist further from the
body and does not know which hand holds the sabre, so a left-hander or a
moment with both arms out will fool it.

---

## The benchmark

```bash
python3 benchmark.py --detect
```

Everything above was measured against a fixed set of real clips, so a
change can be shown to help rather than assumed to.

| Case | Venue | What makes it hard |
|---|---|---|
| portland-still | Portland C2 | the easy one, steady camera |
| portland-fleche | Portland C2 | fencers cross in a fleche |
| seattle-lowres | Seattle D2 | 960x544, fencers about 150px tall, spectator in shot |
| sf-blur-posters | SF AFM | heavy blur, and large photos of fencers on the back wall |
| cincy-pan-blur | Cincinnati NAC | fast hand-held pan, fencers smeared across frames |

The clips are my own competition recordings and are not in the repo. Point
`FENCING_VIDEOS` at your own footage, or edit `CASES`.

There is no automatic ground truth, so each run writes a filmstrip per case
and I decide by eye whether each box is still on the right person. The
printed numbers, losses and warnings, are what I watch for regressions. A
run with no warnings can still be wrong, and a warning can be a false
alarm.

---

## The web page

```bash
python3 webapp.py
```

Then open http://127.0.0.1:8765.

It runs locally. Nothing is uploaded anywhere, there is no account and no
cloud, and it binds to 127.0.0.1 so only that computer can reach it. It
adds no new pip dependencies, just Python's own `http.server`.

Dropping a file copies it into a working folder. Competition footage
routinely runs to 300MB, and copying achieves nothing when the file is
already on the same disk, so there is a path box as well. In Finder,
right-click the file and hold Option, then Copy as Pathname. Quotes and `~`
both work.

The scrubber matters because real clips are whole bouts, 70 seconds of
which four are interesting. Scrub to the phrase you care about before
drawing the boxes, and set Frames to process to cover just that exchange.

The page shells out to the same `main.py` the terminal uses, with the same
defaults, so a result in the browser reproduces on the command line and any
improvement helps both. There is no second implementation to drift out of
step.

---

## Output CSV

One row per frame. Positions are left blank when a fencer is lost, rather
than repeating a stale box that would look like a fresh measurement.

| Column | Meaning |
|---|---|
| `frame_index`, `timestamp_ms` | Real timestamp from the file, not `frame / 30` |
| `camera_dx_px`, `camera_dy_px`, `camera_rotation_deg`, `camera_scale` | This frame's camera movement |
| `camera_estimate_reliable`, `camera_points_used` | Whether to trust it, and how many background points backed it |
| `box_overlap_iou`, `overlap_warning` | How much the two boxes overlap, and whether that has been sustained |
| `fencer_*_tracked` | Did the tracker report success. Not the same as being on the right person |
| `fencer_*_x/y/w/h`, `fencer_*_center_x/y` | The box as seen on screen |
| `fencer_*_stable_x/y` | Position with camera movement removed. Use these for analysis |
| `fencer_*_speed_px_s` | Speed along the piste, positive toward the opponent |
| `fencer_*_movement` | Forward, Backward, Still or Lost |
| `fencer_*_score` | Real confidence, only when the tracker provides one. Blank for CSRT |
| `fencer_*_feet_x/y`, `fencer_*_hip_x/y`, `fencer_*_shoulder_x/y` | Landmarks, with `--pose` |
| `fencer_*_sword_wrist_x/y`, `fencer_*_arm_extension` | Sword hand and reach, with `--pose` |
| `fencer_*_box_jump_px` | How far the box moved this frame. A measured distance, not a confidence |

CSRT does not expose a confidence, so that column stays blank for it rather
than being filled with a number I made up.

---

## What the movement labels mean

Forward means the box moved toward the other fencer along the piste, faster
than a threshold. That is all. It does not identify an attack, a parry, a
riposte, a hit, right of way or who won. A parry is a blade action and this
never looks at the blade.

Three choices sit behind the labels. The piste direction is measured from
the line between the two boxes rather than assumed to be the image's
x-axis, so filming from a corner or in portrait still works. Speed is
measured over about 0.1 seconds instead of one frame, because single-frame
differences on 1080p are mostly noise. A label has to hold for 2 frames
before it switches, since a fencer cannot reverse direction in 33
milliseconds.

The Still threshold is a fraction of the gap between the two fencers when
you selected them, not of the box height. It used to be box height, which
broke as soon as I changed the advice to torso boxes, because a smaller box
halved the threshold and the labels started flickering. The value, 0.05 of
the starting gap or about 63 px/sec on my footage, is hand-tuned and not
calibrated against a real record of when advances happened.

Everything is in pixels. I do not claim real-world distance or speed, which
would need camera calibration and a known reference length. A 14-metre
piste is a good reference to calibrate against later.

---

## Files

| File | Job |
|---|---|
| `video_io.py` | Opening video, measuring real timestamps |
| `tracking.py` | Tracker wrapper, one independent instance per fencer |
| `detect.py` | Person detection, box-to-body matching, momentum, the lost-fencer gate |
| `camera.py` | Camera slide, rotation and zoom |
| `pose.py` | Body landmarks, sword hand, the foot-height check |
| `motion.py` | Piste direction and movement labels |
| `verify.py` | The overlap check |
| `export.py` | Video writer and CSV writer |
| `main.py` | Selection, tracking loop, live window |
| `webapp.py` | The local web page |
| `benchmark.py` | The five-competition test set |
| `get_models.py` | Fetches the three pretrained models |
| `selftest.py` | 53 checks |
| `FILMING.md` | How to film so this works |
| `NOTICE.md` | Third-party code and model licences |

---

## Two bugs worth recording

`cv2.CAP_PROP_FPS` reported 28.20 fps on my first clip, and the real rate
is 29.97. My first version fed that straight into the video writer, so the
export came out 4.256 seconds instead of 4.003, which is 6.3% slow motion.
The frame count in the header was wrong too, claiming 127 frames when only
120 decode. Everything now measures the real timestamps instead of trusting
the header.

The camera correction had its sign backwards at one point, so it doubled
shake instead of removing it. The first clip barely shakes, so nothing
looked wrong in the output. A test against a known synthetic shift caught
it.

---

## Limitations

- A crossing is still the hardest moment, and worth checking by eye even
  though momentum now carries identities through it.
- CSRT can drift onto a similar-looking person and still report success.
  Detection catches most of that, but not a drift onto a real person
  standing where the fencer was predicted.
- Nothing here tells my fencers apart from every other fencer in the hall.
- The box lags during the fastest part of a lunge.
- Blade actions are not recoverable at 30fps.
- Five clips is not a large sample.

## Next steps

1. Film a batch following [FILMING.md](FILMING.md) and add the good ones to
   the benchmark, so the next change is measured against real footage.
2. Shoot slow motion and check whether blade detection becomes realistic.
3. Work out whether pose can drive the tracker without breaking the fencer
   it currently helps least.
4. Longer term, this is stage one of an iPhone app for reviewing bouts.

---

## Licence

MIT, see [LICENSE](LICENSE). Third-party code and the pretrained models
carry their own terms, listed in [NOTICE.md](NOTICE.md).
