# How to film so the tracking works

Every number here comes from measuring real footage. Five segments from
four competitions were run through the tracker, and the filming
conditions were measured frame by frame and lined up against what
actually happened.

| test clip | camera rotation | zoom drift | pan speed | sharpness | what happened |
|---|---|---|---|---|---|
| Portland (steady) | 0.06 °/s | 0.1% | **5.5 px/s** | **566** | **both fencers tracked** |
| Portland (fleche) | 0.64 °/s | **12.2%** | 40.6 px/s | 468 | Fencer A lost late |
| Seattle D2 | 0.52 °/s | 0.1% | 41.6 px/s | 947 | B drifted onto a spectator |
| SF AFM | 0.23 °/s | **7.5%** | **93.6 px/s** | 420 | both failed |
| Cincinnati NAC | 0.10 °/s | 0.3% | 3.2 px/s | **420** | A failed |

One clip tracked both fencers cleanly. It is also the only one filmed
with a pan under 10 px/s and no zoom. That is the whole guide in one
sentence, but the details below are worth reading once.

---

## The five rules, in order of how much they matter

### 1. Do not pan. Put the phone down.

The only fully successful clip had the camera moving 5.5 pixels per
second. The two worst had 41 and 94. Panning does three bad things at
once: it smears the image, it drags every reference point across the
frame, and it changes which people are in shot mid-point.

**Do this:** prop the phone on something and leave it alone for the whole
point. A tripod, a gear bag, a chair, the strip barrier. Frame the piste
once, wide enough that neither fencer can leave it, and do not touch it.

A slightly-too-wide shot that never moves beats a perfectly framed one
that follows the action. Every time.

### 2. Never zoom during a point.

Zoom drift of 7.5% and 12.2% both showed up in failures. Zero zoom
showed up in the success.

Pinch-zooming mid-point is the worst thing you can do, but "cinematic"
auto-zoom counts too. Set your framing before "Fence" and leave it.

### 3. Both fencers must stay in frame for the whole point.

On the SF clip one fencer walks out of shot entirely. Nothing can track
someone who is not there — the software now says "LOST" instead of
pretending, but that point is simply unusable.

**Do this:** frame the full length of the action, including the retreat.
If you are filming from the side, get the whole piste in. Leave more room
than looks necessary — a fleche covers a lot of ground fast.

### 4. Get light on the piste, or shoot slow motion.

Sharpness scored 566 on the clip that worked and 420 on two that failed.
That is motion blur, and it is a shutter-speed problem, not a software
problem.

**Do this:**
- Film the best-lit strip available. Finals strips are usually brightest.
- Better: **shoot in slo-mo, 120fps or 240fps.** Most phones do 1080p at
  both. Shorter exposure per frame means a sharper fencer, and 4-8x more
  samples through the action. This is also the only thing that would make
  blade tracking realistic — at 30fps the blade simply is not recorded
  during fast actions.

### 5. Keep the foreground clear.

On the Seattle clip, Fencer B's box slid onto a **spectator's head** in
the foreground. The software cannot object, because a spectator is a
perfectly real person.

**Do this:** stand where nobody is between you and the piste — no
referee's back, no coach, no other fencers warming up. Slightly elevated
helps a lot: a bench, a step, the second row of seating.

---

## Framing checklist

Run through this once before you press record.

- [ ] Phone is **propped up and will not move**
- [ ] Filming from the **side of the piste**, not from either end
- [ ] Roughly level with the fencers' chests, or a little above
- [ ] **Whole piste in frame**, with room at both ends
- [ ] Fencers fill roughly **half the frame height** (all test clips did,
      including the ones that worked)
- [ ] **Nobody between you and the piste**
- [ ] Zoom set before "Fence" and not touched
- [ ] Slo-mo on if the hall is dim

## Settings

| Setting | Use | Why |
|---|---|---|
| Resolution | **1080p is plenty** | The Seattle test clip is only 960x544 and the person detector still found the fencers on **100%** of frames — the best score of any clip. Resolution is not the bottleneck. Do not bother with 4K. |
| Frame rate | 60fps normally, **120 or 240 for blade work** | Sharper fencers, more samples through fast actions |
| Stabilisation | Leave it on | The Portland clip's stabilisation removed the camera shake entirely — 0.05° of rotation across 4 seconds |
| Format | HEVC is fine | Both HEVC and H.264 clips opened without trouble |

## What still will not work, however you film

- **Two fencers crossing in a fleche.** Momentum now carries their
  identities through the pass, and they no longer collapse onto one
  person — but it is still the hardest moment and worth checking by eye.
- **A fencer passing behind someone in identical kit.** Nothing here
  tells your fencers apart from every other fencer in the hall.
- **Blade actions at 30fps.** Not a software limit. Shoot slo-mo.

## When you do film a batch

Add a good one to the benchmark so improvements can be measured against
real footage rather than guessed at. Open `benchmark.py`, add a line to
`CASES` with the clip path, the start frame, and a torso box for each
fencer, then:

```bash
python3 benchmark.py
```
