# Fencing Video Tracker

Track two hand-picked fencers through competition video. Produces an
annotated clip, a CSV of positions and speeds, optional body landmarks,
and -- the part that took the most work -- an honest answer when it has
lost them.

Built and measured against real footage from five competitions: four
venues, fencers from 100 to 450 pixels tall, one clip at 960x544, heavy
motion blur, hand-held panning, and a back wall covered in photographs of
other fencers.

Python and OpenCV, two pip packages, no cloud, no accounts. Runs from the
terminal or from a local web page.

**Two things worth reading before the rest:**
[FILMING.md](FILMING.md), because how you film matters more than anything
in this code, and
[what I tried that did not work](#what-i-tried-that-did-not-work).

---

## Filming

Before filming a competition, read **[FILMING.md](FILMING.md)**. It is
built from measurements of the test footage: the only segment that
tracked both fencers cleanly is also the only one filmed with a pan under
10 px/s and no zoom. Framing matters more than any setting in this code.

---

## Read this first: how to draw the box

**Box the MASK AND TORSO ONLY. Not the legs. Not the blade.**

This is the single biggest thing you control, and it is worth more than
every parameter in the code. Measured on the `portland-fleche` test clip:

| What you box | What happened |
|---|---|
| Whole fencer (270x450) | Drifted onto **the referee in the blue suit** at frame 49, then tracked him for the remaining 170 frames — reporting success the whole time |
| Mask + torso (205x240) | Followed the fencer correctly through the lunge and the retreat |

Why: a full-body box is mostly empty floor (the gap between the legs,
especially in a lunge), and legs swing around wildly. The tracker ends up
learning "pale wooden floor", which matches floor everywhere. The torso
and mask are compact, solid, and stay the same shape.

The program warns you if the box you draw looks like a full body.

---

## Two ways to run it

**A web page** -- `python3 webapp.py`, then open http://127.0.0.1:8765.
Drop in a clip, scrub to the action, drag two boxes, press Run.

**The terminal** -- everything below. Same code, same results.

---

## Setup

```bash
git clone https://github.com/matthewyongenwang-coder/fencing-video-tracker.git
cd fencing-video-tracker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 get_models.py
```

Two pip packages, and `get_models.py` fetches ~42MB of pretrained models
from the OpenCV Model Zoo. They are not in the repo: they never change and
would bloat every clone. Nothing here is trained by this project.

Plain tracking works with no models at all; you need them for `--detect`
(strongly recommended) and `--pose`.

```bash
python3 main.py "~/Videos/Fencing/IMG_1135 3.MOV"
```

**Keep the quotes** if the path has spaces in it. Inside Python a path
with spaces is just a string; quoting is only a shell requirement.

### Controls

| Key | Does |
|---|---|
| `SPACE` | pause / resume |
| `A` | pause and re-select Fencer A |
| `B` | pause and re-select Fencer B |
| `Q` or `ESC` | stop early (everything so far is still saved) |

Pauses automatically on a reported loss, and on an overlap warning.

### Flags

```bash
--box-a 255,425,205,240 --box-b 1490,400,200,210   # skip the picker
--no-display                                       # batch, no window
--display-width 960                                # smaller window
--tracker vit                                      # see "Two trackers" below
--pose                                             # body landmarks, see below
--detect                                           # RECOMMENDED on real footage
```

### Check nothing is broken

```bash
python3 selftest.py "/path/to/IMG_1135 3.MOV"
```

53 checks, all passing.

---

## When it goes wrong, this is why

Early versions lost fencers most often on retreats and when background
fencers were nearby. Both were reproduced on the `portland-fleche` clip
and traced to specific causes.

### The main failure: the fencers cross

The clip ends in a fleche — the fencers run past each other. Frames
92–100 have the two boxes sitting on top of each other (up to 79%
overlap). Coming out the other side, **Fencer A's tracker picked the
wrong person** and spent the rest of the clip on background fencers and
then empty floor. Fencer B stayed correct.

This is not "the two trackers swapped with each other" — that still
cannot happen, they share nothing. It is one tracker latching onto
whichever body happened to be under its box at the moment of the pass.

It is genuinely close to a coin flip. Running the same clip with the same
starting boxes but a one-frame offset made the tracker follow the *other*
fencer out of the crossing.

### The second failure: silent drift

Worse than losing the fencer is **not** losing them. On the full-body run
Fencer A's box slid onto the referee at frame 49 and CSRT reported
success for 170 straight frames. Nothing in the output said anything was
wrong. That is why the live window matters — the CSV alone will lie to
you.

---

## What I tried that did not work

Being explicit, because it explains why there is only one automatic
check in the code rather than five.

| Idea | Why it failed |
|---|---|
| Cap how far a box may jump per frame | A **correct** box during a fast lunge jumped 0.284 x its own height. The **drifting** box jumped 0.356. Too close. And the other drift only jumped 0.194 — *less* than the correct lunge. Any threshold either misses real drift or fires on real lunges. |
| Compare tracker motion against optical flow of the box contents | Correct tracking during a blurry lunge disagreed by 116px; an actual drift by 157px. Overlapping again. |
| Raise CSRT's own `psr_threshold` so it gives up sooner | At 0.06, no change at all. At 0.10 it declared failure at frame 30 — in the middle of a perfectly good lunge. |
| Other CSRT tuning (`filter_lr`, `padding`, `template_size`, `use_segmentation`) | Swept all of them. Every variant still ended up on the referee. |

Tuning does not fix this. The box you draw does.

---

## The one automatic check that does work

**Two fencers cannot be in the same place.** If both boxes sit on top of
each other for a sustained stretch, at least one is wrong. That is
geometry — it does not care about lighting, blur, or everyone wearing
white.

On the failing clip it fired at frame 99, which is exactly where Fencer
A went wrong.

It allows brief overlap, because a fleche really does send fencers past
each other. Only a sustained overlap (8+ frames at 45%+) is reported.

**It will sometimes cry wolf** on a genuinely close pass where nothing
broke. The cost is one pause and a glance. When it fires, the useful move
is usually: press `SPACE` to carry on a few frames until the fencers
separate, *then* press `A` or `B` to re-select cleanly — trying to draw a
box while they are on top of each other is hard.

**What it does NOT catch:** one tracker drifting onto a background person
while the other stays correct — the referee case at frame 49 produced no
overlap at all. You still have to watch the window. I would rather ship
one check that works than four that produce false alarms you learn to
ignore.

---

## Two trackers to choose from

```bash
python3 main.py "clip.MOV"                  # csrt, the default
python3 main.py "clip.MOV" --tracker vit    # needs models/vittrack.onnx
```

Measured on `IMG_1135 4.MOV`, Fencer A:

| | CSRT | ViT |
|---|---|---|
| Holds through blur | Better | Gives up sooner (lost by frame 30) |
| When it goes wrong | **Silently** — tracked the referee for 170 frames claiming success | **Honestly** — reports the loss, score collapses from 0.44 to 0.09 |
| Confidence score | None. CSRT does not expose one, so the CSV column stays blank rather than being filled with an invented number | Real: `getTrackingScore()`, written to the CSV |
| Extra files | None | One 715KB ONNX model |

Neither is simply better. CSRT is the default because it is stickier and
needs no download. ViT is worth trying when you would rather be told
"I lost them" than be quietly misled.

ViT is a pretrained model released by OpenCV — using an existing tool,
not training anything. It is already downloaded in `models/`. If you ever
need it again:

```bash
curl -L -o models/vittrack.onnx https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx
```

---

## Camera movement

The camera correction now measures **slide, rotation and zoom** together,
not just sliding. That changed because your two clips are completely
different:

| | rotation | zoom | drift |
|---|---|---|---|
| `IMG_1135 3.MOV` (still) | 0.05° | 0.9994x | 7 px |
| `IMG_1135 4.MOV` (panning) | 1.01° | **1.0924x** | 164 px |

A 9.2% zoom is not a rounding error, and sliding-only correction ignored
it entirely. The steady clip barely moves at all — phone stabilisation had
already done the work there, so the blur in fast actions is the *fencers*
moving, not the camera.

Still not modelled: perspective and parallax. The floor stretches away
from the camera, so near things shift more than far things — one
transform cannot be right for both depths. Over a long clip the running
transform also accumulates drift.

---

## The web page

```bash
cd ~/fencing-tracker && source venv/bin/activate
python3 webapp.py
```

Then open **http://127.0.0.1:8765**.

Drop a clip in (or paste its path), drag the scrubber to the action, draw
a box round each fencer, press Run. You get a progress bar, any warnings
as they happen, the annotated video playing in the page, and the CSV to
download.

**It runs locally.** Nothing is uploaded anywhere, no account, no
cloud. It binds to 127.0.0.1, so only this computer can reach it -- not
other machines on the wifi. It adds **no new pip dependencies**: Python's
own `http.server` and nothing else.

### Paste the path for big clips

Dropping a file copies it into a working folder. Competition footage
routinely runs to 300MB, and copying achieves nothing when the file is
already on the same disk. So there is a path box: in Finder, right-click the file and hold
Option, then "Copy as Pathname". Quotes and `~` are both handled.

### The scrubber matters

Your real clips are whole bouts -- 70 seconds of which four are
interesting. Scrub to the phrase you care about before drawing the boxes,
and set "Frames to process" to cover just that exchange. Tracking a
fencer standing still for a minute tells you nothing and takes a while.

### It is the same code

The page shells out to the same `main.py` you run from Terminal, with the
same defaults. That is deliberate: a result you get in the browser is one
you can reproduce on the command line, and any improvement helps both.
There is no second implementation to drift out of step.

The two flags it exposes are the two that matter: **Person detection**
(on by default, see above) and **Body landmarks**.

`--start-frame`, `--max-frames` and `--progress` were added to `main.py`
for this, and they are useful from the terminal too:

```bash
python3 main.py "bout.MOV" --start-frame 1950 --max-frames 150 --detect
```


---

## Tested on real footage from five competitions

Everything before this was tuned on one 4-second clip with a steady
camera and a clean angle. Real fencing video is blurrier, more
off-angle and faster than that, so the benchmark below was built from
footage across four venues:

```bash
python3 benchmark.py --detect
```

| case | venue | what makes it hard |
|---|---|---|
| portland-still | Portland C2 | the easy one, steady camera |
| portland-fleche | Portland C2 | fencers cross in a fleche |
| seattle-lowres | Seattle D2 | **960x544**, fencers ~150px tall, spectator in shot |
| sf-blur-posters | SF AFM | heavy blur, and **giant photos of fencers on the back wall** |
| cincy-pan-blur | Cincinnati NAC | fast hand-held pan, fencers smeared across frames |

### The baseline was bad, and it was bad in one specific way

Of 10 fencer-tracks, only **5 were still on the right person** at the
end. Every single failure landed on the same kind of thing:

| case | what the box ended up on |
|---|---|
| portland-fleche A | a referee standing still |
| seattle-lowres B | a spectator's dark hair |
| sf-blur-posters A | the ceiling edge |
| sf-blur-posters B | the scoreboard |
| cincy-pan-blur A | a sponsor banner |

A banner, a scoreboard and a ceiling edge are static and high-contrast,
and CSRT prefers them to a blurred fencer mid-lunge. Template tracking
has no idea what a person is, so nothing stops this.

**And it never said a word.** All five reported `tracked=True` the whole
time.

### The fix: only track things that are people

```bash
python3 main.py "clip.MOV" --detect
```

YOLOX (36MB model, no new pip install, run through cv2.dnn) finds every
person in each frame. A fencer's box must contain a detected body. If it
does not, for 12 frames straight, that fencer is declared **LOST**
instead of left sitting on scenery.

This removes the failure mode structurally rather than trying to spot it
afterwards. It cannot drift onto a banner because a banner is not a
person.

Verified on the hard footage: it finds the fencers even when badly
smeared — 0.84 confidence on the heavily blurred SF fencer, 0.68 on the
motion-streaked Cincinnati one. It does **not** report the giant
photographs of fencers on the SF back wall as people, which was the thing
I most expected to go wrong.

Result: **four of the five failures are now flagged** instead of silent.
On portland-fleche, Fencer A now ends the clip marked LOST in red rather
than confidently tracking a referee.

There is also mutual exclusion in the matching: one detected body can be
claimed by at most one fencer. That structurally prevents both boxes
collapsing onto the same person, which the overlap warning could
previously only complain about after the fact.

### What it still does not fix

**A box drifting onto a different PERSON.** On the Seattle clip a box
slid onto a spectator's head, and a spectator is a perfectly good person,
so nothing objects. Telling *your* fencers apart from everyone else in
the hall is a separate problem and is not solved.

**Crossings.** When two fencers genuinely run past each other, detection
gives you two bodies and no way to know which is which coming out.

**A fencer who leaves the frame.** On the SF clip the left fencer walks
out of shot entirely. Nothing can track someone who is not there — but
now it says "lost" instead of pretending.

### Recommended settings for real footage

```bash
python3 main.py "clip.MOV" --detect --pose
```

`--detect` is the one that matters. `--pose` adds the skeleton, the feet,
and the sword hand. Together they run at roughly a third of real time on
this Mac, which is fine for clips of a few seconds.


---

## Crossings: momentum, not appearance

When two fencers run past each other in a fleche, both boxes sit over the
same patch of image for several frames. A template tracker has no way to
tell which body is which coming out, and it was close to a coin flip --
the same clip with a one-frame offset sent the tracker after the *other*
fencer.

But a fleche is not ambiguous physically. **Both fencers are carrying
momentum in opposite directions, and nobody reverses instantly mid-pass.**
So the fix is to stop asking "which body looks like the one I remember"
(they are identical, in identical kit) and start asking "which body is
where this fencer must have gone".

Each fencer now carries a velocity, measured in the camera-stabilised
frame so a pan cannot be mistaken for sprinting. Detections are matched
against a predicted position as well as the last known box, and when
nobody can be matched the tracker coasts on momentum for up to half a
second before giving up.

Measured on the fleche clip:

| | before | after |
|---|---|---|
| Peak overlap between the two boxes | collapsed onto one person | **0.24, never merged** |
| Fencer A at the end | silently tracking a referee at x=1275 | **declared LOST** |
| Identity through the pass | coin flip | both boxes on distinct, correct fencers at f105, f140, f165 (checked by eye) |

One implementation note worth keeping: matching against the prediction
*alone* made things worse. On a clip that previously tracked perfectly,
box jitter produced noisy velocity, the prediction overshot, matching
failed, and the fencer coasted further off -- a feedback loop that cost 24
frames on an easy clip. Matching against **both** the current box and the
prediction, and taking whichever fits better, fixed that: the current box
anchors ordinary frames, and the prediction only earns its keep during a
crossing.

The detector's patience also had to go up from 12 frames to 20. Recall on
a clip that tracks perfectly is about 94% -- the misses are the deep lunge
frames, where a fully extended fencer stops looking like the upright
person the detector expects. Twelve frames of patience turned that into a
false "LOST" on an easy clip.

### Still not solved

- A box drifting onto a different **person** who happens to be where the
  fencer was predicted. Momentum narrows the field; it does not close it.
- Telling *your* fencers apart from every other fencer in the hall.
- A fencer who leaves the frame entirely. Now honestly reported as lost.


---

## Body landmarks (the "lines on the body")

```bash
python3 main.py "clip.MOV" --pose
```

This is BlazePose, a small pretrained model from Google that OpenCV
publishes. It draws 33 body points per fencer: shoulders, elbows,
wrists, hips, knees, ankles. **No new pip install** -- it is a 5MB model
file run through OpenCV's own DNN module, the same arrangement as the
ViT tracker.

It is pointed at each fencer's tracked box rather than at the whole
frame. That matters: the skeleton then always belongs to the fencer
*you* selected, instead of whichever body a detector happened to rank
first in a hall full of identical white kit. It also means you only need
to box the torso -- the model still finds the legs and feet below it.

### What it gives you that a box does not

- **Ankles.** Where the fencer actually meets the piste. A box edge is
  arbitrary; feet are real.
- **Wrists and elbows.** The sword hand and how extended the arm is.
- **A real "is anyone there?" answer.** Point it at empty floor and it
  returns nothing at all.

### The foot-height check

This is the strongest drift detector in the project so far, and it only
became possible once we had feet.

A fencer on a piste has feet that move smoothly and stay low in frame. A
tracker that has wandered onto someone in the background lands on feet
that are suddenly much higher up, because distant people's feet sit
higher. Measured on the clip where Fencer A drifted:

| frames | Fencer A feet y | Fencer B feet y |
|---|---|---|
| 0–24 | ~795 (correct) | ~725 (correct) |
| 108–216 | **~500–600** (background) | ~660 (still correct) |

A rose ~290px and stayed there. B moved ~65px across the whole clip.
The check fires on A at frames 125 and 158, and stays silent on B and on
both fencers of the clip that tracks correctly.

Note what does **not** work: pose *confidence*. It sat at 0.97+ the whole
time A was tracking the wrong people — because those people are
perfectly real bodies. It is the **height** of the feet that gives it
away, not whether a body was found.

Two bugs this went through, both instructive:

1. The first version compared feet against a fast-adapting running
   average — which simply followed the drift down, so the gap always
   looked small and it never fired. A reference you are measuring
   movement *away from* must not chase that movement.
2. It also cleared its evidence every frame the pose model found
   nothing. Pose only lands on ~75% of frames here (it fails on the
   blurriest ones), so the gaps wiped the count before it could ever
   trigger. A missing measurement is not evidence that things are fine.

### Pose does NOT drive the tracker, and here is why

The obvious next move is to let pose re-lock the box each frame. I tried
it, twice. On clip 4, Fencer A:

| | end of clip, centre x | truth |
|---|---|---|
| CSRT alone | 1179 (lost, on background) | ~100–350 |
| CSRT + pose re-lock | **205 (correct)** | ~100–350 |

That is a broken track turned into a correct one. But the *same* change
made Fencer B worse — B was tracking correctly and the pose lock pushed
it off. A conservative version that only intervenes on disagreement
rejected 180 of 221 frames for B as implausible, which stripped CSRT of
its stability without giving anything back.

So it is a real lead, not a finished feature, and it is not wired in.
Pose currently **measures and warns**; it does not steer. Turning it into
a net win needs a better plausibility test and a way to correct the box
without destroying the tracker's history.

---

## Blades: the honest answer

A reasonable hope for this project is blade tracking. The pixels were
checked before answering.

**When a fencer is still, the blade is visible** — a dark line about 2–3
pixels wide, with the guard clearly readable.

**During exactly the actions you care about, it is not there.** At frame
64 of clip 1 the blade has smeared into the background and only the
guard survives as a grey blur. At frame 88, mid-lunge, the whole arm is a
streak and the blade is essentially gone.

That is not a limitation of the algorithm. It is the footage: 30fps with
a phone's automatic shutter cannot freeze a sabre blade. No detector can
find something the sensor did not record.

**What would actually fix it: film in slow motion.** A modern phone shoots
1080p at 120 or 240fps. That is 4–8x shorter exposure per frame and 4–8x
more samples through the action. It is by far the highest-value change
you could make, and it costs nothing but a settings toggle.

**What you can measure today, without the blade:** the sword hand and
arm extension, from the pose landmarks. `fencer_*_sword_wrist_x/y`,
`fencer_*_sword_elbow_x/y` and `fencer_*_arm_extension` are in the CSV.
Arm extension is reach from shoulders to sword hand, in torso-widths, so
it does not change just because the fencer moved further from the
camera.

Be clear about what that is and is not. It tells you the arm extended.
It does **not** tell you there was an attack, a parry, or a hit — and in
sabre, right of way turns on exactly those distinctions. The sword-hand
guess is also just geometry: it picks the wrist further from the body and
does not know which hand holds the sabre, so a left-hander or a moment
with both arms out will fool it.


---

## What was tested

**Verified on real footage:**

- Both clips open (HEVC / Dolby Vision is not a problem).
- Export plays: decodes back frame-for-frame, macOS AVFoundation reports
  `isPlayable: true` and hardware-decodable, and QuickTime Player opened
  the file and held a read handle on it.
- Export timing matches the source: 4.003s in, 4.003s out.
- Clip 1 tracks correctly end to end — checked by eye at 11 points, not
  just by the absence of errors.
- Clip 4 fails as described above — also checked by eye.
- Camera correction recovers known synthetic slides, a 1.5° rotation and
  a 5% zoom, and undoes each for a stationary point.
- Mid-clip re-selection: the self-test wrecks the tracker at frame 40,
  rescues it at frame 60 the way pressing `A` does, and it converges back
  to within 2px of the clean run.
- The A / B keys work interactively — you confirmed that.

**Still not verified:**

- **QuickTime playing it visually.** AVFoundation decoded it and
  QuickTime opened it, but an automation permission dialog blocked the
  script from clicking Play. Double-click the file; it takes two seconds.
- Any clip other than these two.

---

## Output CSV

Positions are blank when a fencer is lost, rather than repeating a stale
box that would look like a fresh measurement.

| Column | Meaning |
|---|---|
| `frame_index`, `timestamp_ms` | Real timestamp from the file, not `frame / 30`. |
| `camera_dx_px`, `camera_dy_px`, `camera_rotation_deg`, `camera_scale` | This frame's camera movement. |
| `camera_estimate_reliable`, `camera_points_used` | Whether to trust it, and how many background points backed it. |
| `box_overlap_iou`, `overlap_warning` | How much the two boxes overlap, and whether that has been sustained. |
| `fencer_*_tracked` | Did the tracker report success. **Not** the same as "is on the right person". |
| `fencer_*_x/y/w/h`, `fencer_*_center_x/y` | The box as seen on screen. |
| `fencer_*_stable_x/y` | Position with camera movement removed. **Use these** for analysis and the robot project. |
| `fencer_*_speed_px_s` | Speed along the piste, positive = toward the opponent. Pixels per second. |
| `fencer_*_movement` | Forward / Backward / Still / Lost. |
| `fencer_*_score` | Real confidence — only when the tracker provides one. Blank for CSRT. |
| `fencer_*_box_jump_px` | How far the box moved this frame. A measured distance, not a confidence. |

---

## What the movement labels do and do not say

"Forward" means the box moved toward the other fencer along the piste,
faster than a threshold. Nothing more. It does **not** identify an
attack, parry, riposte, hit, right of way, or who won. A parry is a blade
action, and this prototype never looks at the blade.

Design choices:

1. **Piste direction is measured**, from the line between your two boxes
   — not assumed to be the image's x-axis. Works if you film from a
   corner or in portrait.
2. **Speed over a ~0.1s window**, not one frame. Single-frame differences
   on 1080p are mostly noise.
3. **A label must hold 2 frames before it switches.** A fencer cannot
   reverse direction in 33ms.
4. **The "Still" threshold is a fraction of the gap between the two
   fencers at selection**, not of the box height. It used to be box
   height, which silently broke when the advice changed to torso boxes —
   a smaller box halved the threshold and the labels started flickering.

The threshold (0.05 of the starting gap, about 63 px/sec here) is
**hand-tuned**, not calibrated against a real ground truth of when
advances happened.

Everything is in pixels. No real-world distance or speed is claimed —
that would need camera calibration and a known reference length. A
14-metre piste is a good future reference.

---

## Files

| File | Job |
|---|---|
| `video_io.py` | Opening the video, measuring real timestamps |
| `tracking.py` | Tracker wrapper, one independent instance per fencer |
| `camera.py` | Camera slide / rotation / zoom, so it isn't counted as fencer movement |
| `motion.py` | Piste direction and movement labels |
| `verify.py` | The overlap check |
| `detect.py` | Person detection, box-to-body matching, the lost-fencer gate |
| `benchmark.py` | The five-competition test set |
| `webapp.py` | The local web page (stdlib only, no new dependencies) |
| `get_models.py` | Fetches the three pretrained models |
| `FILMING.md` | **How to film so this works** -- read it before your next comp |
| `export.py` | Video writer (codec choice) and CSV writer |
| `main.py` | Selection, tracking loop, live window |
| `pose.py` | Body landmarks, the sword hand, the foot-height check |
| `vendor/` | Third-party pose preprocessing from the OpenCV Zoo (Apache 2.0) |
| `selftest.py` | The 53 checks |

---

## Honest limitations

- **A crossing is close to a coin flip.** If the fencers pass each other,
  expect to re-select. Nothing in this version fixes that.
- **CSRT can drift and still report success.** Watch the window.
- **The overlap check misses single-tracker drift**, and can cry wolf on
  a close pass.
- **The box lags during the fastest part of a lunge.**
- **Two clips is not a sample.** Nothing here is proven to generalise.

## Sensible next steps

1. Use torso boxes from now on and see how much that alone helps across
   more clips.
2. When it fails, note whether it was a crossing or a background drift.
   Those need different fixes.
3. The real fix for crossings is a person detector plus re-identification
   — deciding *which* of the two bodies is which after they separate.
   That is a much bigger step, and it is the right one to take once you
   have enough clips to test against.
4. Blade detection is what any actual refereeing help would need, and
   none of this touches it yet.
