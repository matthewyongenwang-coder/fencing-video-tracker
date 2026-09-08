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
 * CAMERA CORRECTION IS HERE, BUT ONLY PARTLY
 * camera.py fits slide, rotation and zoom with optical flow and RANSAC.
 * Doing that properly here means shipping OpenCV.js at 8MB, so CameraShift
 * at the bottom of this file estimates sliding only, with block matching
 * on a downscaled greyscale copy. That covers the dominant term. Rotation
 * and zoom are still uncorrected, which is one more reason to read
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
  // maxMove may be one number for everyone, or one per fencer. Per fencer
  // matters because a fencer we have not seen for a while could be
  // anywhere within a widening circle, while one we saw last frame could
  // not have gone far.
  const limitFor = i => Array.isArray(maxMove) ? maxMove[i] : maxMove;
  const candidates = [];
  current.forEach((box, fi) => {
    if (!box) return;
    const options = [box];
    if (predicted && predicted[fi]) options.push(predicted[fi]);

    detections.forEach((det, di) => {
      let best = null;
      const limit = limitFor(fi);
      for (const option of options) {
        const inside = containment(option, det.box);
        const torso = torsoFromPerson(det.box, { w: option.w, h: option.h });
        const move = Math.hypot(
          torso.x + torso.w / 2 - (option.x + option.w / 2),
          torso.y + torso.h / 2 - (option.y + option.h / 2));
        if (move > limit) continue;

        // Two ways to accept a detection, and it needs only one.
        //
        // Containment is the strict one: the box we are holding is mostly
        // inside this person. That is the right test when tracking is
        // going well.
        //
        // It is the wrong test during a fast pass. The box lags the
        // fencer by a frame or two, containment drops under the
        // threshold, and the fencer is dropped while standing in plain
        // sight. Measured on the fleche clip: Fencer A stopped matching
        // any of THIRTEEN detected people, coasted until it ran out of
        // patience, and was declared lost. The fencers were 522px apart
        // at the time, so nothing was merged. The match was just too
        // fussy.
        //
        // So also accept a detection whose torso lands close to where
        // this fencer is or is predicted to be. `move` is already capped
        // above, so this cannot reach across the hall.
        const near = move < 0.6 * option.h;
        if (inside < 0.45 && !near) continue;

        // Containment still scores higher, so a confident overlap beats a
        // merely nearby body when both are on offer.
        const score = Math.max(inside, near ? 0.45 : 0) + 0.3 * det.score
                      - (move / Math.max(limit, 1)) * 0.5;
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
  constructor({ giveUpAfter = 20, resnapBelowIou = 0.55, maxCoastFrames = 15,
                maxMergeFrames = 60 } = {}) {
    // 20 frames is about 0.7 seconds. It was 12 in an earlier version and
    // that was too impatient: detector recall on a clip that tracks
    // perfectly is about 94%, and the misses are the deep lunge frames
    // where a fully extended fencer stops looking like the upright person
    // the detector expects.
    this.giveUpAfter = giveUpAfter;
    this.resnapBelowIou = resnapBelowIou;
    this.maxCoastFrames = maxCoastFrames;
    // How long to keep coasting when the two fencers are inside the SAME
    // detection. A fleche pass regularly takes longer than half a second,
    // and going LOST in the middle of one is the wrong answer: we know
    // exactly where both fencers are, they are just in the same box.
    this.maxMergeFrames = maxMergeFrames;
    this.motion = [new FencerMotion(), new FencerMotion()];
    this.noBody = [0, 0];
    this.lost = [false, false];
    this.recoverFrames = [0, 0];
    this.lastPerson = [null, null];
    // How far from the last known position we will look when trying to
    // pick a lost fencer back up, in multiples of their box height.
    this.recoverRadius = 3.0;
  }

  /**
   * Look for a lost fencer near where they were last seen.
   *
   * Deliberately strict about size. A fencer who has walked out of shot
   * must stay lost rather than being replaced by a spectator who happens
   * to be standing in roughly the right place.
   */
  _reacquire(i, fencers, detections, claimed) {
    const remembered = this.lastPerson[i];
    const box = fencers[i].box;
    if (!remembered) return null;
    const other = fencers[1 - i];
    const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
    const radius = this.recoverRadius * box.h;
    let best = null, bestDistance = Infinity;
    detections.forEach((det, di) => {
      if (claimed.has(di)) return;
      const ratio = det.box.h / remembered.h;
      if (ratio < 0.6 || ratio > 1.7) return;
      // Never recover onto the person the other fencer is already on.
      // Checking the claimed set is not enough, because if the other
      // fencer is coasting their detection was never claimed, and this is
      // exactly how both boxes end up on one fencer.
      if (other.tracked) {
        const theirs = torsoFromPerson(det.box, { w: other.box.w, h: other.box.h });
        if (intersectionOverUnion(theirs, other.box) > 0.30) return;
      }
      const d = Math.hypot(det.box.x + det.box.w / 2 - cx,
                           det.box.y + det.box.h * 0.30 - cy);
      if (d > radius || d >= bestDistance) return;
      bestDistance = d; best = det.box;
    });
    return best;
  }

  /**
   * `stable` converts between screen coordinates and coordinates with the
   * camera's movement taken out. Momentum has to be measured with the
   * camera removed, because in raw screen coordinates a pan looks exactly
   * like a fencer sprinting. Pass null to skip the correction.
   */
  step(fencers, detections, dt, stable) {
    const toStable = p => stable ? stable.stabilise(p) : p;
    const toScreen = p => stable ? stable.toImage(p) : p;

    const predicted = fencers.map((f, i) => {
      const guess = this.motion[i].predict(dt);
      if (!guess) return null;
      const onScreen = toScreen(guess);
      return { x: Math.round(onScreen.x - f.box.w / 2),
               y: Math.round(onScreen.y - f.box.h / 2), w: f.box.w, h: f.box.h };
    });

    const current = fencers.map(f => f.box);

    // How far each fencer is allowed to have moved since we last had a
    // fix on them. This widens the longer they have been coasting.
    //
    // A fixed cap is wrong, and it is what broke the crossing. Momentum
    // predicts a straight line, and a lunge is not a straight line: the
    // fencer drives forward, then reverses and recovers. On the fleche
    // clip Fencer A's box carried on rightward at x=565 while the real
    // fencer had turned and gone back to x=333. That is 230 pixels, the
    // cap was 108, so the correct body was rejected on every frame and
    // the fencer was declared lost while standing in plain sight.
    //
    // Widening with time is the standard answer: the longer since a real
    // observation, the less we know, so the further we should be willing
    // to look. It stays bounded, so it never becomes "match anyone".
    const maxMove = fencers.map((f, i) => {
      const coasted = this.motion[i].coasting;
      const base = 0.45 * Math.max(...current.map(b => b.h));
      return base * Math.min(1 + coasted * 0.30, 4.0);
    });
    const matched = matchDetections(current, detections, maxMove, predicted);
    const claimed = new Set();
    matched.forEach(m => { if (m) claimed.add(detections.findIndex(d => d.box === m)); });

    // ARE THEY CROSSING?
    //
    // When two fencers run past each other the detector stops seeing two
    // people and returns one box round both of them. Only one fencer can
    // claim it, so the other has nothing to match, coasts, runs out of
    // patience and is declared lost. That is what actually broke the
    // crossing here: not the matching, the giving up.
    //
    // A merge is easy to recognise. One detection contains both fencers'
    // centres, and it is wider than one fencer has any business being.
    // While that is true, an unmatched fencer is not missing. We know
    // where they are, they are in that box, so keep coasting on momentum
    // and do not start the clock on declaring them lost.
    const centreOf = b => ({ x: b.x + b.w / 2, y: b.y + b.h / 2 });
    const inside = (p, b) => p.x >= b.x && p.x <= b.x + b.w &&
                             p.y >= b.y && p.y <= b.y + b.h;
    const centres = fencers.map((f, i) => centreOf(predicted[i] || f.box));
    const merging = detections.some(d =>
      centres.every(c => inside(c, d.box)) &&
      d.box.w > 1.4 * Math.max(...fencers.map(f => f.box.w)));

    return fencers.map((fencer, i) => {
      const wasLost = this.lost[i];
      const person = matched[i];
      const info = { matched: !!person, snapped: false, lost: wasLost,
                     newlyLost: false, recovered: false, coasting: 0,
                     merging: false, speed: this.motion[i].speed() };

      if (!person) {
        // Already given up on this one? Try to pick them up again.
        //
        // Without this, a fencer lost at a crossing stays lost for the
        // rest of the clip, because matching needs the stale box to
        // overlap a detection and the stale box is nowhere near anybody.
        // Recovery looks near where they were last seen, for someone of
        // about the right size who is not already claimed, and insists on
        // seeing them twice before believing it.
        if (wasLost) {
          const found = this._reacquire(i, fencers, detections, claimed);
          if (found) {
            this.recoverFrames[i] += 1;
            if (this.recoverFrames[i] >= 2) {
              fencer.box = torsoFromPerson(found, { w: fencer.box.w, h: fencer.box.h });
              fencer.tracked = true;
              claimed.add(detections.findIndex(d => d.box === found));
              this.noBody[i] = 0;
              this.lost[i] = false;
              this.recoverFrames[i] = 0;
              this.motion[i] = new FencerMotion();
              this.motion[i].observe(toStable({
                x: fencer.box.x + fencer.box.w / 2,
                y: fencer.box.y + fencer.box.h / 2 }), dt, fencer.box.h);
              info.lost = false;
              info.recovered = true;
              return info;
            }
          } else {
            this.recoverFrames[i] = 0;
          }
        }

        // Only start counting toward "lost" if this is a genuine
        // disappearance rather than a crossing.
        if (!merging) this.noBody[i] += 1;
        info.merging = merging;

        const coastLimit = merging ? this.maxMergeFrames : this.maxCoastFrames;
        if (this.motion[i].position && this.motion[i].coasting < coastLimit) {
          const c = this.motion[i].coast(dt);
          const onScreen = toScreen(c);
          fencer.box = { x: Math.round(onScreen.x - fencer.box.w / 2),
                         y: Math.round(onScreen.y - fencer.box.h / 2),
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
      this.lastPerson[i] = { h: person.h, w: person.w };
      this.motion[i].observe(
        toStable({ x: fencer.box.x + fencer.box.w / 2,
                   y: fencer.box.y + fencer.box.h / 2 }),
        dt, fencer.box.h);
      info.speed = this.motion[i].speed();
      return info;
    });
  }

  reset(index) {
    this.motion[index] = new FencerMotion();
    this.noBody[index] = 0;
    this.lost[index] = false;
    this.recoverFrames[index] = 0;
    this.lastPerson[index] = null;
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

// ----------------------------------------------------------------------
// Camera movement, without shipping OpenCV
// ----------------------------------------------------------------------

/**
 * Estimates how far the camera moved between two frames.
 *
 * WHY THIS IS HERE NOW
 * The first browser version skipped camera correction, on the grounds
 * that doing it properly means shipping OpenCV.js at 8MB. That was a
 * mistake on panning footage. The momentum model works on velocities, and
 * in raw screen coordinates a pan looks exactly like a fencer sprinting.
 * On the fleche test clip the camera pans 164px and zooms 9.2%, so every
 * prediction going into the crossing was wrong by that much.
 *
 * HOW IT WORKS
 * Take a grid of small patches from the previous frame, skipping anything
 * near a fencer, and find where each one moved to by searching a small
 * window in the new frame. Score by sum of absolute differences, which is
 * cheap and good enough for whole-image motion. The median of all the
 * patch displacements is the camera's movement, and a median means a few
 * patches that landed on a moving person cannot drag the answer around.
 *
 * All of it happens on a downscaled greyscale copy, so a 1920x1080 frame
 * becomes 320x180 and the whole thing costs about a millisecond.
 *
 * WHAT IT DOES NOT DO
 * Only sliding. Not rotation, not zoom. The Python version fits a full
 * similarity transform with RANSAC. This handles the dominant term and
 * leaves the rest, which is another reason FILMING.md says do not pan.
 */
export class CameraShift {
  constructor(width = 320) {
    this.width = width;
    this.prev = null;
    this.scale = 1;
    this.canvas = null;
    this.ctx = null;
  }

  _grey(source) {
    if (!this.canvas) {
      this.canvas = new OffscreenCanvas(this.width, 1);
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    }
    const h = Math.round(this.width * source.height / source.width);
    if (this.canvas.height !== h) this.canvas.height = h;
    this.scale = source.width / this.width;
    this.ctx.drawImage(source, 0, 0, this.width, h);
    const rgba = this.ctx.getImageData(0, 0, this.width, h).data;
    const grey = new Uint8Array(this.width * h);
    for (let i = 0; i < grey.length; i++) {
      // luma, near enough
      grey[i] = (rgba[i * 4] * 77 + rgba[i * 4 + 1] * 150 + rgba[i * 4 + 2] * 29) >> 8;
    }
    return { data: grey, w: this.width, h };
  }

  /** Returns { dx, dy, reliable, points } in FULL resolution pixels. */
  update(source, fencerBoxes) {
    const cur = this._grey(source);
    if (!this.prev || this.prev.w !== cur.w || this.prev.h !== cur.h) {
      this.prev = cur;
      return { dx: 0, dy: 0, reliable: false, points: 0 };
    }
    const prev = this.prev;
    const PATCH = 8, SEARCH = 10, STEP = 20;
    // Fencer boxes in small-image coordinates, grown a little so a blade
    // or a trailing foot just outside the box is skipped too.
    const skip = fencerBoxes.filter(Boolean).map(b => ({
      x: b.x / this.scale - 8, y: b.y / this.scale - 8,
      w: b.w / this.scale + 16, h: b.h / this.scale + 16,
    }));
    const inSkip = (x, y) => skip.some(s =>
      x >= s.x && x <= s.x + s.w && y >= s.y && y <= s.y + s.h);

    const dxs = [], dys = [];
    for (let y = SEARCH + PATCH; y < cur.h - SEARCH - PATCH; y += STEP) {
      for (let x = SEARCH + PATCH; x < cur.w - SEARCH - PATCH; x += STEP) {
        if (inSkip(x, y)) continue;
        // Skip flat patches. A blank stretch of floor matches everywhere
        // equally well and contributes nothing but noise.
        let mn = 255, mx = 0;
        for (let j = 0; j < PATCH; j += 2) {
          for (let i = 0; i < PATCH; i += 2) {
            const v = prev.data[(y + j) * prev.w + (x + i)];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
          }
        }
        if (mx - mn < 18) continue;

        let bestScore = Infinity, bdx = 0, bdy = 0;
        for (let oy = -SEARCH; oy <= SEARCH; oy += 2) {
          for (let ox = -SEARCH; ox <= SEARCH; ox += 2) {
            let sad = 0;
            for (let j = 0; j < PATCH; j += 2) {
              const pr = (y + j) * prev.w + x;
              const cr = (y + j + oy) * cur.w + x + ox;
              for (let i = 0; i < PATCH; i += 2) {
                sad += Math.abs(prev.data[pr + i] - cur.data[cr + i]);
              }
            }
            if (sad < bestScore) { bestScore = sad; bdx = ox; bdy = oy; }
          }
        }
        dxs.push(bdx); dys.push(bdy);
      }
    }

    this.prev = cur;
    if (dxs.length < 12) return { dx: 0, dy: 0, reliable: false, points: dxs.length };

    const median = a => { const s = [...a].sort((p, q) => p - q); return s[s.length >> 1]; };
    // The patches moved with the background. The camera moved the other
    // way, and the answer is wanted in full resolution pixels.
    return { dx: -median(dxs) * this.scale, dy: -median(dys) * this.scale,
             reliable: true, points: dxs.length };
  }
}

/**
 * Keeps a running total of camera movement so a position can be converted
 * into "where it would be if the camera had never moved".
 */
export class StableFrame {
  constructor() { this.dx = 0; this.dy = 0; }
  accumulate(shift) { if (shift.reliable) { this.dx += shift.dx; this.dy += shift.dy; } }
  stabilise(p) { return { x: p.x + this.dx, y: p.y + this.dy }; }
  toImage(p) { return { x: p.x - this.dx, y: p.y - this.dy }; }
}
