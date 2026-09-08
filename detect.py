"""
detect.py

Finds PEOPLE in each frame, so the tracker has real candidates to snap to
instead of drifting onto whatever texture happens to look similar.

WHY THIS EXISTS, and the measurement that forced it
Running the old tracker over five clips from four competitions, only 5 of
10 fencer-tracks were still on the right person at the end. Every single
failure ended up on the same kind of thing:

    portland-fleche   Fencer A -> a referee standing still
    seattle-lowres    Fencer B -> a spectator's dark hair
    sf-blur-posters   Fencer A -> the ceiling edge
                      Fencer B -> the scoreboard
    cincy-pan-blur    Fencer A -> a sponsor banner

A banner, a scoreboard and a ceiling edge have one thing in common: they
are static, high-contrast, and CSRT likes them more than a blurred fencer
mid-lunge. Template tracking has no concept of "person", so nothing stops
this.

A person detector does. YOLOX cannot return a banner, because a banner is
not in its vocabulary. That removes the whole failure mode structurally
rather than trying to detect it after the fact, which I tried four
different ways and could not make separate reliably (see the README).

WHAT IT IS
YOLOX-S, released by OpenCV in their Model Zoo, run through cv2.dnn. No
new pip dependency, same arrangement as the pose and ViT models. 36MB.

Measured on this Mac: 37 fps at 416x416 input, 18 fps at 640x640.

VERIFIED ON THE HARD FOOTAGE
It finds the fencers even when they are badly smeared: 0.84 confidence on
the heavily blurred fencer in the San Francisco clip, 0.68 on the
motion-streaked one in Cincinnati.

IT DOES FIRE ON THE WALL POSTERS, though. The San Francisco venue has
large printed photographs of fencers behind the piste, and the detector
reports them as people at 0.72 to 0.85 confidence. Across that 150 frame
segment, 15% of frames come back with more detections than there are real
people in shot.

I first checked a single frame, saw no poster detections, and wrote that
it was not a problem. That was wrong, and it is a good reminder that one
frame is not a measurement.

It does not break tracking, because match_detections() below needs the
fencer's existing box to overlap the detection AND the movement to be
continuous AND the detection to be unclaimed. A printed fencer is static
and far from where a real fencer was predicted, so nothing ever matches
it. Detections are candidates, not conclusions.

WHAT IT STILL CANNOT DO
It finds people. It does not know WHICH person is Fencer A, because every
fencer, referee, coach and spectator comes back as an equally valid
person. Deciding which detection belongs to which fencer is the
association step below, and that is still just geometry and motion
continuity. When two fencers genuinely cross, this does not magically
solve it.
"""

import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "yolox_person.onnx")

# COCO class 0 is "person". We ignore the other 79.
PERSON_CLASS = 0

# YOLOX predicts on three grids, at these strides.
STRIDES = (8, 16, 32)


def _build_grid(size):
    """Anchor centres and their strides, in model-input pixels."""
    grids, strides = [], []
    for stride in STRIDES:
        cells = size // stride
        ys, xs = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
        grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
        strides.append(np.full((cells * cells, 1), stride))
    return np.concatenate(grids, 0), np.concatenate(strides, 0)


class PersonDetector:
    """
    Every person in the frame, as (x, y, w, h) boxes plus a confidence
    that the model genuinely provides.
    """

    def __init__(self, input_size=640, confidence=0.25, nms_threshold=0.50):
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Person detector not found at {MODEL_PATH}\n"
                "Download it once with:\n"
                "  mkdir -p models && curl -L -o models/yolox_person.onnx \\\n"
                "    https://github.com/opencv/opencv_zoo/raw/main/models/"
                "object_detection_yolox/object_detection_yolox_2022nov.onnx\n"
                "Or run without --detect.")
        self.net = cv2.dnn.readNet(MODEL_PATH)
        self.size = input_size
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.grid, self.grid_strides = _build_grid(input_size)

    def detect(self, frame):
        """Returns [((x, y, w, h), score), ...], biggest-scoring first."""
        height, width = frame.shape[:2]
        # Letterbox: scale to fit, pad the rest, so nothing is squashed.
        scale = min(self.size / width, self.size / height)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized

        self.net.setInput(cv2.dnn.blobFromImage(canvas))
        raw = self.net.forward()[0]                     # (anchors, 85)

        # YOLOX predicts offsets from each grid cell, and log-scale sizes.
        centres = (raw[:, :2] + self.grid) * self.grid_strides
        sizes = np.exp(raw[:, 2:4]) * self.grid_strides
        # objectness x class probability
        scores = (raw[:, 4:5] * raw[:, 5:])[:, PERSON_CLASS]

        keep = scores > self.confidence
        if not keep.any():
            return []

        boxes = np.concatenate([centres[keep] - sizes[keep] / 2,
                                sizes[keep]], 1) / scale
        kept_scores = scores[keep]

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), kept_scores.tolist(),
                                   self.confidence, self.nms_threshold)
        if len(indices) == 0:
            return []

        results = [(tuple(int(v) for v in boxes[i]), float(kept_scores[i]))
                   for i in np.array(indices).ravel()]
        results.sort(key=lambda item: -item[1])
        return results


def torso_from_person(person_box, keep_size=None):
    """
    Turn a whole-person detection into the mask-and-torso box this project
    tracks.

    The torso sits in the upper-middle of a standing person, so the centre
    goes about 30% of the way down. If `keep_size` is given we reuse the
    tracker's existing box size instead of recomputing it. That stops the
    box pulsing in size every time the detector wobbles, which was what
    made an earlier pose-based version collapse to a tiny square.
    """
    x, y, w, h = person_box
    centre_x = x + w / 2.0
    centre_y = y + h * 0.30
    if keep_size is not None:
        box_w, box_h = keep_size
    else:
        box_w, box_h = w * 0.80, h * 0.42
    return (int(centre_x - box_w / 2), int(centre_y - box_h / 2),
            int(box_w), int(box_h))


def _containment(inner, outer):
    """How much of `inner` lies inside `outer`, 0.0 to 1.0."""
    ax, ay, aw, ah = inner
    bx, by, bw, bh = outer
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    return ((right - left) * (bottom - top)) / float(max(aw * ah, 1))


def match_detections(fencer_boxes, detections, max_move, predicted_boxes=None):
    """
    Decide which detection, if any, belongs to each fencer.

    Two rules do the work:

    1. CONTINUITY. A fencer cannot teleport, so a detection only counts if
       the fencer's current box is substantially inside it and its centre
       has not moved further than `max_move` since last frame.

    2. MUTUAL EXCLUSION. One detection can be claimed by at most one
       fencer. This is worth more than it looks: it structurally prevents
       both boxes collapsing onto the same person, which is the failure
       the overlap warning could previously only complain about after the
       fact.

    Matching is greedy, best score first, which is enough for two targets.

    Returns a list the same length as fencer_boxes, holding either the
    matched detection box or None.
    """
    candidates = []
    for f_index, f_box in enumerate(fencer_boxes):
        if f_box is None:
            continue
        # Two places a fencer might reasonably be: exactly where the
        # tracker last had them, and where their momentum says they
        # should have got to. Score against BOTH and keep the better.
        #
        # Using the prediction ALONE was tried and made things worse: on
        # a clip that previously tracked perfectly, box jitter produced
        # noisy velocity, the prediction overshot, the match failed, and
        # the fencer coasted further off, a feedback loop that lost 24
        # frames on an easy clip. The current box keeps it anchored on
        # ordinary frames; the prediction only earns its keep during a
        # crossing, when the current box is on top of the wrong person.
        options = [f_box]
        if predicted_boxes is not None and predicted_boxes[f_index] is not None:
            options.append(predicted_boxes[f_index])

        for d_index, (d_box, d_score) in enumerate(detections):
            best = None
            for option in options:
                inside = _containment(option, d_box)
                if inside < 0.45:
                    continue
                o_cx = option[0] + option[2] / 2.0
                o_cy = option[1] + option[3] / 2.0
                torso = torso_from_person(
                    d_box, keep_size=(option[2], option[3]))
                move = np.hypot(torso[0] + torso[2] / 2.0 - o_cx,
                                torso[1] + torso[3] / 2.0 - o_cy)
                if move > max_move:
                    continue
                # Prefer lots of overlap, a confident detection, little movement.
                score = inside + 0.3 * d_score - (move / max(max_move, 1.0)) * 0.5
                if best is None or score > best:
                    best = score
            if best is not None:
                candidates.append((best, f_index, d_index))

    candidates.sort(reverse=True)
    assigned = [None] * len(fencer_boxes)
    used_fencers, used_detections = set(), set()
    for _score, f_index, d_index in candidates:
        if f_index in used_fencers or d_index in used_detections:
            continue
        used_fencers.add(f_index)
        used_detections.add(d_index)
        assigned[f_index] = detections[d_index][0]
    return assigned


class FencerMotion:
    """
    Where a fencer is heading, measured in the steady reference frame.

    WHY THIS EXISTS, the crossing problem
    When two fencers run past each other in a fleche, both boxes end up
    over the same patch of image for several frames. Coming out the other
    side, a template tracker has no way to tell which body is which, and
    measurement showed it was close to a coin flip: the same clip with a
    one-frame offset sent the tracker after the OTHER fencer.

    But a fleche is not ambiguous physically. Both fencers are carrying
    real momentum in opposite directions. Nobody reverses direction
    instantly in the middle of a pass. So if we know Fencer A was moving
    right at 600 px/sec going in, then coming out, the body still moving
    right is Fencer A.

    That is the whole idea: stop asking "which body looks like the one I
    remember" (they are identical) and start asking "which body is where
    this fencer must have gone" (they are not).

    WHY THE STEADY FRAME AND NOT THE IMAGE
    Velocity has to be measured with the camera's own movement removed.
    In raw image coordinates a hand-held pan looks exactly like the fencer
    sprinting, so predictions would be dragged along by the camera. All
    the maths here happens in camera.stabilise() coordinates, and only the
    final answer is converted back to the image.
    """

    def __init__(self, smoothing=0.6, max_speed_factor=6.0):
        self.position = None            # stabilised (x, y)
        self.velocity = np.zeros(2)     # stabilised pixels per second
        self.smoothing = smoothing
        # Speed cap as a multiple of box heights per second. A fencer
        # covers a few body-heights a second at most; anything faster is
        # a tracking glitch and must not be extrapolated.
        self.max_speed_factor = max_speed_factor
        self.coasting = 0

    def observe(self, stabilised_point, dt, box_height):
        """Feed in a trusted position and update the velocity estimate."""
        point = np.array(stabilised_point, dtype=float)
        if self.position is not None and dt > 0:
            measured = (point - self.position) / dt
            cap = self.max_speed_factor * max(box_height, 1)
            speed = float(np.linalg.norm(measured))
            if speed > cap:
                measured = measured * (cap / speed)
            self.velocity = (self.smoothing * self.velocity +
                             (1.0 - self.smoothing) * measured)
        self.position = point
        self.coasting = 0

    def predict(self, dt):
        """Where the fencer should be after dt seconds, if nothing changed."""
        if self.position is None:
            return None
        return tuple(self.position + self.velocity * dt)

    def coast(self, dt):
        """
        Carry on under momentum because we could not see them this frame.

        Velocity is bled off as we coast: the longer we go without a real
        observation, the less we should trust that they kept going. After
        a while the prediction stops moving rather than flying off across
        the hall.
        """
        if self.position is None:
            return None
        self.position = self.position + self.velocity * dt
        self.velocity *= 0.90
        self.coasting += 1
        return tuple(self.position)

    def speed(self):
        return float(np.linalg.norm(self.velocity))


class DetectionGate:
    """
    Keeps each fencer's box attached to an actual detected person, decides
    WHICH person that is using momentum, and says so when it cannot find
    them at all.

    THREE RULES DO THE WORK

    1. IT MUST BE A PERSON. If no detected body is in a fencer's box for
       long enough, that fencer is declared LOST rather than left sitting
       on a sponsor banner. Measured across five real clips, this flagged
       four of the five known failures, which previously all reported
       success.

    2. IT MUST BE WHERE THEY WERE GOING. Detections are matched against a
       PREDICTED position from the fencer's own velocity, not against
       where the box happened to be last frame. This is what survives a
       fleche: both fencers keep their momentum through the pass, so the
       body still travelling right is still Fencer A.

    3. ONE BODY, ONE FENCER. A detection can be claimed by at most one
       fencer, so both boxes can never collapse onto the same person.

    WHAT IT STILL MISSES
    A box drifting onto a different PERSON who happens to be where the
    fencer was predicted to be. A spectator standing in the right place at
    the right moment will be accepted. Momentum narrows the field a lot;
    it does not close it.
    """

    def __init__(self, detector=None, give_up_after=20, resnap_below_iou=0.55,
                 max_coast_frames=15):
        self.detector = detector or PersonDetector()
        # 20 frames is about 0.7 seconds.
        #
        # This was 12, and 12 was too impatient. Measured detector recall
        # on a clip that tracks perfectly is about 94%, and the misses are
        # the deep lunge frames, where a fully extended fencer stops
        # looking like the upright person the detector expects. Twelve
        # frames of patience turned that into a false "LOST" on an easy
        # clip. Twenty rides out the lunge while still catching a real
        # loss inside a second.
        self.give_up_after = give_up_after
        self.resnap_below_iou = resnap_below_iou
        # How long we will follow momentum alone with nobody matched.
        # Half a second: enough to carry through a fleche pass, short
        # enough that a genuinely lost fencer is admitted quickly.
        self.max_coast_frames = max_coast_frames
        self.motion = {}
        self.no_body = {}
        self.lost = {}

    def _state(self, fencer):
        key = id(fencer)
        if key not in self.motion:
            self.motion[key] = FencerMotion()
        return key

    def step(self, frame, fencers, camera, dt):
        """
        Update every fencer against this frame's detections.

        `camera` is the CameraMotion instance and `dt` the seconds since
        the previous frame. Both are needed because prediction happens in
        the steady reference frame, not in image coordinates.
        """
        from verify import intersection_over_union

        people = self.detector.detect(frame)

        # Where do we EXPECT each fencer to be? Match against that, not
        # against last frame's box.
        predicted_boxes = []
        for fencer in fencers:
            key = self._state(fencer)
            motion = self.motion[key]
            x, y, w, h = fencer.box
            guess = motion.predict(dt)
            if guess is None:
                predicted_boxes.append(fencer.box)
                continue
            cx, cy = camera.to_image(guess)
            predicted_boxes.append((int(cx - w / 2), int(cy - h / 2), w, h))

        current_boxes = [f.box for f in fencers]
        heights = [b[3] for b in current_boxes] or [100]
        max_move = 0.45 * max(heights)
        matched = match_detections(current_boxes, people, max_move,
                                   predicted_boxes=predicted_boxes)

        report = []
        for index, (fencer, person) in enumerate(zip(fencers, matched)):
            key = self._state(fencer)
            motion = self.motion[key]
            was_lost = self.lost.get(key, False)
            entry = {"matched": person is not None, "snapped": False,
                     "lost": was_lost, "newly_lost": False, "recovered": False,
                     "coasting": 0, "speed": motion.speed(),
                     "people_seen": len(people)}

            if person is None:
                self.no_body[key] = self.no_body.get(key, 0) + 1
                # Carry on under momentum for a short while. This is what
                # gets a fencer through a crossing, where the detector may
                # merge two bodies into one box for several frames.
                if motion.position is not None and \
                        motion.coasting < self.max_coast_frames:
                    coasted = motion.coast(dt)
                    cx, cy = camera.to_image(coasted)
                    x, y, w, h = fencer.box
                    fencer.box = (int(cx - w / 2), int(cy - h / 2), w, h)
                    entry["coasting"] = motion.coasting
                if self.no_body[key] >= self.give_up_after:
                    fencer.tracked = False
                    entry["lost"] = True
                    entry["newly_lost"] = not was_lost
                    self.lost[key] = True
                report.append(entry)
                continue

            self.no_body[key] = 0
            fencer.tracked = True
            entry["lost"] = False
            entry["recovered"] = was_lost
            self.lost[key] = False

            snapped = torso_from_person(
                person, keep_size=(fencer.box[2], fencer.box[3]))
            if intersection_over_union(snapped, fencer.box) < self.resnap_below_iou:
                fencer.reinit(frame, snapped)
                fencer.tracked = True
                entry["snapped"] = True

            centre = (fencer.box[0] + fencer.box[2] / 2.0,
                      fencer.box[1] + fencer.box[3] / 2.0)
            motion.observe(camera.stabilise(centre), dt, fencer.box[3])
            entry["speed"] = motion.speed()
            report.append(entry)

        return report

    def reset(self, fencer):
        key = id(fencer)
        self.motion[key] = FencerMotion()
        self.no_body[key] = 0
        self.lost[key] = False
