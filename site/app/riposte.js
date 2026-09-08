/*
 * riposte.js
 *
 * The tracking logic, ported from the Python version to run in a browser.
 *
 * WHAT IS HERE AND WHAT IS NOT
 * Everything in this file is arithmetic: matching detections to fencers,
 * predicting where a fencer should be, deciding when one is lost, and
 * labelling movement along the piste. All of it is a direct port of
 * detect.py, verify.py and motion.py.
 *
 * CSRT IS GONE ON PURPOSE
 * The Python version runs an OpenCV CSRT tracker per fencer, and there is
 * no browser equivalent. Before writing this I removed CSRT from the
 * Python version and re-ran the five clip benchmark. It worked where the
 * original worked and failed where the original already failed, so the
 * tracker was doing less than it looked. Detection every frame plus the
 * momentum model below covers the same job.
 *
 * CAMERA CORRECTION IS ALSO GONE, FOR NOW
 * camera.py estimates slide, rotation and zoom with optical flow. Doing
 * that here means shipping OpenCV.js, which is an 8MB download, so v1
 * skips it. Everything below treats screen coordinates as stable
 * coordinates. On a clip filmed from a tripod that is true. On a panning
 * clip the momentum model will be wrong, which is one more reason to read
 * FILMING.md and put the phone down.
 */

// ----------------------------------------------------------------------
// Geometry
// ----------------------------------------------------------------------

export function intersectionOverUnion(a, b) {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.w, b.x + b.w);
  const bottom = Math.min(a.y + a.h, b.y + b.h);
  if (right <= left || bottom <= top) return 0;
  const overlap = (right - left) * (bottom - top);
  const union = a.w * a.h + b.w * b.h - overlap;
  return union > 0 ? overlap / union : 0;
}

/** How much of `inner` lies inside `outer`, 0 to 1. */
function containment(inner, outer) {
  const left = Math.max(inner.x, outer.x);
  const top = Math.max(inner.y, outer.y);
  const right = Math.min(inner.x + inner.w, outer.x + outer.w);
  const bottom = Math.min(inner.y + inner.h, outer.y + outer.h);
  if (right <= left || bottom <= top) return 0;
  return ((right - left) * (bottom - top)) / Math.max(inner.w * inner.h, 1);
}

/**
 * Turn a whole person detection into the mask and torso box we track.
 *
 * The torso sits in the upper middle of a standing person, so the centre
 * goes about 30% of the way down. Passing keepSize reuses the box size we
 * already have, which stops the box pulsing every time the detector
 * wobbles.
 */
export function torsoFromPerson(person, keepSize) {
  const cx = person.x + person.w / 2;
  const cy = person.y + person.h * 0.30;
  const w = keepSize ? keepSize.w : person.w * 0.80;
  const h = keepSize ? keepSize.h : person.h * 0.42;
  return { x: Math.round(cx - w / 2), y: Math.round(cy - h / 2),
           w: Math.round(w), h: Math.round(h) };
}

/**
 * Decide which detection, if any, belongs to each fencer.
 *
 * Two rules do the work. A detection only counts if the fencer's box is
 * already substantially inside it and it has not moved further than
 * maxMove, which is continuity. And one detection can be claimed by at
 * most one fencer, which is what stops both boxes collapsing onto the
 * same person.
 *
 * Each fencer is scored against both its current box and its predicted
 * position, keeping whichever fits better. Prediction alone was tried in
 * the Python version and made things worse: box jitter produced noisy
 * velocity, the prediction overshot, and the fencer coasted further off.
 */
export function matchDetections(current, detections, maxMove, predicted) {
  const candidates = [];
  current.forEach((box, fi) => {
    if (!box) return;
    const options = [box];
    if (predicted && predicted[fi]) options.push(predicted[fi]);

    detections.forEach((det, di) => {
      let best = null;
      for (const option of options) {
        const inside = containment(option, det.box);
        if (inside < 0.45) continue;
        const torso = torsoFromPerson(det.box, { w: option.w, h: option.h });
        const move = Math.hypot(
          torso.x + torso.w / 2 - (option.x + option.w / 2),
          torso.y + torso.h / 2 - (option.y + option.h / 2));
        if (move > maxMove) continue;
        const score = inside + 0.3 * det.score - (move / Math.max(maxMove, 1)) * 0.5;
        if (best === null || score > best) best = score;
      }
      if (best !== null) candidates.push([best, fi, di]);
    });
  });

  candidates.sort((a, b) => b[0] - a[0]);
  const assigned = current.map(() => null);
  const usedFencers = new Set(), usedDetections = new Set();
  for (const [, fi, di] of candidates) {
    if (usedFencers.has(fi) || usedDetections.has(di)) continue;
    usedFencers.add(fi);
    usedDetections.add(di);
    assigned[fi] = detections[di].box;
  }
  return assigned;
}

// ----------------------------------------------------------------------
// Momentum, which is how identity survives a crossing
// ----------------------------------------------------------------------

/**
 * Where a fencer is heading.
 *
 * Two fencers in identical kit passing each other look the same, so
 * appearance cannot separate them. A fleche is not ambiguous physically
 * though. Both are carrying momentum in opposite directions and neither
 * reverses instantly mid pass, so the body still travelling right is
 * still Fencer A.
 */
export class FencerMotion {
  constructor(smoothing = 0.6, maxSpeedFactor = 6.0) {
    this.position = null;
    this.velocity = { x: 0, y: 0 };
    this.smoothing = smoothing;
    // Speed cap in box heights per second. A fencer covers a few body
    // heights a second at most, and anything faster is a glitch that must
    // not be extrapolated.
    this.maxSpeedFactor = maxSpeedFactor;
    this.coasting = 0;
  }

  observe(point, dt, boxHeight) {
    if (this.position && dt > 0) {
      let vx = (point.x - this.position.x) / dt;
      let vy = (point.y - this.position.y) / dt;
      const cap = this.maxSpeedFactor * Math.max(boxHeight, 1);
      const speed = Math.hypot(vx, vy);
      if (speed > cap) { vx *= cap / speed; vy *= cap / speed; }
      this.velocity = {
        x: this.smoothing * this.velocity.x + (1 - this.smoothing) * vx,
        y: this.smoothing * this.velocity.y + (1 - this.smoothing) * vy,
      };
    }
    this.position = { x: point.x, y: point.y };
    this.coasting = 0;
  }

  predict(dt) {
    if (!this.position) return null;
    return { x: this.position.x + this.velocity.x * dt,
             y: this.position.y + this.velocity.y * dt };
  }

  /**
   * Carry on under momentum because nobody was matched this frame.
   * Velocity bleeds off as we coast, so a long gap stops the prediction
   * flying off across the hall instead of quietly going nowhere.
   */
  coast(dt) {
    if (!this.position) return null;
    this.position = { x: this.position.x + this.velocity.x * dt,
                      y: this.position.y + this.velocity.y * dt };
    this.velocity = { x: this.velocity.x * 0.90, y: this.velocity.y * 0.90 };
    this.coasting += 1;
    return this.position;
  }

  speed() { return Math.hypot(this.velocity.x, this.velocity.y); }
}

// ----------------------------------------------------------------------
// The gate: a box has to be on a person, or we say we lost them
// ----------------------------------------------------------------------

export class DetectionGate {
  constructor({ giveUpAfter = 20, resnapBelowIou = 0.55, maxCoastFrames = 15 } = {}) {
    // 20 frames is about 0.7 seconds. It was 12 in an earlier version and
    // that was too impatient: detector recall on a clip that tracks
    // perfectly is about 94%, and the misses are the deep lunge frames
    // where a fully extended fencer stops looking like the upright person
    // the detector expects.
    this.giveUpAfter = giveUpAfter;
    this.resnapBelowIou = resnapBelowIou;
    this.maxCoastFrames = maxCoastFrames;
    this.motion = [new FencerMotion(), new FencerMotion()];
    this.noBody = [0, 0];
    this.lost = [false, false];
  }

  step(fencers, detections, dt) {
    const predicted = fencers.map((f, i) => {
      const guess = this.motion[i].predict(dt);
      if (!guess) return null;
      return { x: Math.round(guess.x - f.box.w / 2),
               y: Math.round(guess.y - f.box.h / 2), w: f.box.w, h: f.box.h };
    });

    const current = fencers.map(f => f.box);
    const maxMove = 0.45 * Math.max(...current.map(b => b.h));
    const matched = matchDetections(current, detections, maxMove, predicted);

    return fencers.map((fencer, i) => {
      const wasLost = this.lost[i];
      const person = matched[i];
      const info = { matched: !!person, snapped: false, lost: wasLost,
                     newlyLost: false, recovered: false, coasting: 0,
                     speed: this.motion[i].speed() };

      if (!person) {
        this.noBody[i] += 1;
        // Coast through the gap. This is what gets a fencer through a
        // crossing, where the detector may merge two bodies for a while.
        if (this.motion[i].position && this.motion[i].coasting < this.maxCoastFrames) {
          const c = this.motion[i].coast(dt);
          fencer.box = { x: Math.round(c.x - fencer.box.w / 2),
                         y: Math.round(c.y - fencer.box.h / 2),
                         w: fencer.box.w, h: fencer.box.h };
          info.coasting = this.motion[i].coasting;
        }
        if (this.noBody[i] >= this.giveUpAfter) {
          fencer.tracked = false;
          info.lost = true;
          info.newlyLost = !wasLost;
          this.lost[i] = true;
        }
        return info;
      }

      this.noBody[i] = 0;
      fencer.tracked = true;
      info.lost = false;
      info.recovered = wasLost;
      this.lost[i] = false;

      const snapped = torsoFromPerson(person, { w: fencer.box.w, h: fencer.box.h });
      if (intersectionOverUnion(snapped, fencer.box) < this.resnapBelowIou) {
        fencer.box = snapped;
        info.snapped = true;
      }
      this.motion[i].observe(
        { x: fencer.box.x + fencer.box.w / 2, y: fencer.box.y + fencer.box.h / 2 },
        dt, fencer.box.h);
      info.speed = this.motion[i].speed();
      return info;
    });
  }

  reset(index) {
    this.motion[index] = new FencerMotion();
    this.noBody[index] = 0;
    this.lost[index] = false;
  }
}

// ----------------------------------------------------------------------
// Two fencers cannot be in the same place
// ----------------------------------------------------------------------

export class OverlapWatch {
  constructor(iouThreshold = 0.45, framesBeforeWarning = 8) {
    this.iouThreshold = iouThreshold;
    // 8 frames is about a quarter of a second. A fleche pass clears in
    // less than that. Two boxes stuck on one person do not.
    this.framesBeforeWarning = framesBeforeWarning;
    this.consecutive = 0;
    this.warned = false;
  }

  update(a, b, bothTracked) {
    if (!bothTracked) { this.consecutive = 0; return { iou: 0, warning: false, isNew: false }; }
    const iou = intersectionOverUnion(a, b);
    if (iou >= this.iouThreshold) this.consecutive += 1;
    else { this.consecutive = 0; this.warned = false; }
    const warning = this.consecutive >= this.framesBeforeWarning;
    const isNew = warning && !this.warned;
    if (isNew) this.warned = true;
    return { iou, warning, isNew };
  }
}

// ----------------------------------------------------------------------
// Movement along the piste
// ----------------------------------------------------------------------

/**
 * The direction of the piste, worked out from where the two boxes were
 * placed rather than assumed to be the image's x axis. Filming from a
 * corner or in portrait still works.
 */
export class PisteAxis {
  constructor(centreA, centreB) {
    const vx = centreB.x - centreA.x, vy = centreB.y - centreA.y;
    const length = Math.hypot(vx, vy);
    if (length < 1) throw new Error(
      "The two boxes are on top of each other, so the piste direction cannot be worked out.");
    this.unit = { x: vx / length, y: vy / length };
    this.separation = length;
  }
  project(p) { return p.x * this.unit.x + p.y * this.unit.y; }
}

const STILL_FRACTION_OF_SEPARATION = 0.05;
const SPEED_WINDOW_SEC = 0.10;
const MIN_HOLD_FRAMES = 2;

/**
 * Forward means the box moved toward the other fencer along the piste,
 * faster than a threshold. That is all it means. It does not identify an
 * attack, a parry, a riposte, a hit or right of way. A parry is a blade
 * action and nothing here looks at the blade.
 */
export class MovementLabeler {
  constructor(axis, sign) { this.axis = axis; this.sign = sign; this.reset(); }

  reset() {
    this.history = [];
    this.current = "Still";
    this.pending = null;
    this.pendingFrames = 0;
  }

  debounce(raw) {
    // A fencer cannot reverse direction in 33 milliseconds. Without this
    // the labels flicker on single frames through the most interesting
    // part of a clip.
    if (raw === this.current) { this.pending = null; this.pendingFrames = 0; return this.current; }
    if (raw === this.pending) this.pendingFrames += 1;
    else { this.pending = raw; this.pendingFrames = 1; }
    if (this.pendingFrames >= MIN_HOLD_FRAMES) {
      this.current = raw; this.pending = null; this.pendingFrames = 0;
    }
    return this.current;
  }

  update(timeSec, centre, tracked) {
    if (!tracked) { this.reset(); return { label: "Lost", speed: 0 }; }

    const position = this.axis.project(centre);
    this.history.push([timeSec, position]);
    while (this.history.length > 2 && timeSec - this.history[0][0] > SPEED_WINDOW_SEC) {
      this.history.shift();
    }
    if (this.history.length < 2) return { label: this.debounce("Still"), speed: 0 };

    const [tOld, posOld] = this.history[0];
    const dt = timeSec - tOld;
    if (dt <= 0) return { label: this.debounce("Still"), speed: 0 };

    const speed = ((position - posOld) / dt) * this.sign;
    // The threshold is a fraction of the gap between the two fencers at
    // selection, not of the box height. Box height broke as soon as the
    // advice changed to torso boxes, because a smaller box halved the
    // threshold and the labels started flickering.
    const threshold = STILL_FRACTION_OF_SEPARATION * this.axis.separation;
    const raw = Math.abs(speed) < threshold ? "Still" : (speed > 0 ? "Forward" : "Backward");
    return { label: this.debounce(raw), speed };
  }
}

// ----------------------------------------------------------------------
// CSV, matching the columns the Python version writes
// ----------------------------------------------------------------------

export function buildCsv(rows) {
  const header = [
    "frame_index", "timestamp_ms", "box_overlap_iou", "overlap_warning",
  ];
  for (const tag of ["a", "b"]) {
    header.push(
      `fencer_${tag}_tracked`,
      `fencer_${tag}_x`, `fencer_${tag}_y`, `fencer_${tag}_w`, `fencer_${tag}_h`,
      `fencer_${tag}_center_x`, `fencer_${tag}_center_y`,
      `fencer_${tag}_speed_px_s`, `fencer_${tag}_movement`,
      `fencer_${tag}_detected`);
  }
  const lines = [header.join(",")];
  for (const r of rows) {
    const cells = [r.frame, r.timeMs.toFixed(2), r.iou.toFixed(3), r.overlapWarning];
    for (const f of r.fencers) {
      if (f.tracked) {
        cells.push(true, f.box.x, f.box.y, f.box.w, f.box.h,
                   (f.box.x + f.box.w / 2).toFixed(1),
                   (f.box.y + f.box.h / 2).toFixed(1),
                   f.speed.toFixed(1), f.movement, f.detected);
      } else {
        // Blank rather than a stale box. A stale box that looks like a
        // fresh measurement is how you end up trusting data you should not.
        cells.push(false, "", "", "", "", "", "", "", f.movement, f.detected);
      }
    }
    lines.push(cells.join(","));
  }
  return lines.join("\n");
}
