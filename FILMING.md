# How to film so the tracking works

Every number here comes from measuring real footage. I ran five segments
from four competitions through the tracker, measured the filming conditions
frame by frame, and lined them up against what actually happened.

| Test clip | Camera rotation | Zoom drift | Pan speed | Sharpness | What happened |
|---|---|---|---|---|---|
| Portland, steady | 0.06 °/s | 0.1% | 5.5 px/s | 566 | both fencers tracked |
| Portland, fleche | 0.64 °/s | 12.2% | 40.6 px/s | 468 | Fencer A lost late |
| Seattle D2 | 0.52 °/s | 0.1% | 41.6 px/s | 947 | B drifted onto a spectator |
| SF AFM | 0.23 °/s | 7.5% | 93.6 px/s | 420 | both failed |
| Cincinnati NAC | 0.10 °/s | 0.3% | 3.2 px/s | 420 | A failed |

One clip tracked both fencers cleanly, and it is also the only one filmed
with a pan under 10 px/s and no zoom. That is the whole guide in one
sentence, but the details below are worth reading once.

---

## The five rules, in order of how much they matter

### 1. Do not pan. Put the phone down.

The only fully successful clip had the camera moving 5.5 pixels per second.
The two worst had 41 and 94. Panning smears the image, drags every
reference point across the frame, and changes which people are in shot
mid-point.

Prop the phone on something and leave it alone for the whole point. A
tripod, a gear bag, a chair, the strip barrier. Frame the piste once, wide
enough that neither fencer can leave it, and do not touch it.

A slightly too wide shot that never moves beats a perfectly framed one that
follows the action, every time.

### 2. Never zoom during a point.

Zoom drift of 7.5% and 12.2% both showed up in failures, and zero zoom
showed up in the success.

Pinch-zooming mid-point is the worst version of this, but "cinematic"
auto-zoom counts too. Set your framing before "Fence" and leave it.

### 3. Both fencers must stay in frame for the whole point.

On the SF clip one fencer walks out of shot entirely. Nothing can track
someone who is not there. The software now says LOST instead of pretending,
but that point is unusable.

Frame the full length of the action, including the retreat. If you are
filming from the side, get the whole piste in, and leave more room than
looks necessary. A fleche covers a lot of ground fast.

### 4. Get light on the piste, or shoot slow motion.

Sharpness scored 566 on the clip that worked and 420 on two that failed.
That is motion blur, and it is a shutter-speed problem rather than a
software one.

Film the best-lit strip available, since finals strips are usually
brightest. Better than that, shoot in slo-mo at 120 or 240fps. Most phones
do 1080p at both. Shorter exposure per frame means a sharper fencer and 4
to 8 times more samples through the action. This is also the only thing
that would make blade tracking realistic, because at 30fps the blade is not
recorded during fast actions.

### 5. Keep the foreground clear.

On the Seattle clip, Fencer B's box slid onto a spectator's head in the
foreground. The software cannot object, because a spectator is a real
person.

Stand where nobody is between you and the piste. No referee's back, no
coach, no other fencers warming up. Slightly elevated helps a lot: a bench,
a step, the second row of seating.

---

## Framing checklist

Run through this once before you press record.

- [ ] Phone is propped up and will not move
- [ ] Filming from the side of the piste, not from either end
- [ ] Roughly level with the fencers' chests, or a little above
- [ ] Whole piste in frame, with room at both ends
- [ ] Fencers fill roughly half the frame height, which all my test clips
      did, including the ones that worked
- [ ] Nobody between you and the piste
- [ ] Zoom set before "Fence" and not touched
- [ ] Slo-mo on if the hall is dim

## Settings

| Setting | Use | Why |
|---|---|---|
| Resolution | 1080p is plenty | The Seattle test clip is only 960x544 and the person detector still found the fencers on 100% of frames, the best score of any clip. Resolution is not the bottleneck, so do not bother with 4K. |
| Frame rate | 60fps normally, 120 or 240 for blade work | Sharper fencers, more samples through fast actions |
| Stabilisation | Leave it on | The Portland clip's stabilisation removed the camera shake entirely, leaving 0.05° of rotation across 4 seconds |
| Format | HEVC is fine | Both HEVC and H.264 clips opened without trouble |

## What will not work, however you film

Two fencers crossing in a fleche is still the hardest moment. Momentum
carries their identities through the pass now and they no longer collapse
onto one person, but it is worth checking by eye.

A fencer passing behind someone in identical kit is not solved either.
Nothing here tells my fencers apart from every other fencer in the hall.

Blade actions at 30fps are not a software limit. Shoot slo-mo.

## When you film a batch

Add a good clip to the benchmark so improvements can be measured against
real footage rather than guessed at. Open `benchmark.py`, add a line to
`CASES` with the clip path, the start frame, and a torso box for each
fencer, then run:

```bash
python3 benchmark.py
```
