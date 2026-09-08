"""
verify.py

Watches for signs that a tracker has quietly wandered onto the wrong
person, and says so.

WHY THIS EXISTS
The worst failure in this whole project is not a tracker giving up. It
is a tracker confidently following the wrong person. On one of my test
clips, Fencer A's box slid off the fencer at frame 49 and spent the next
170 frames tracking the referee in the blue suit, reporting success
the entire time. Nothing in the output said anything was wrong.

WHAT I TRIED THAT DID NOT WORK
Being honest about this, because it shapes what's here:

* A limit on how far a box may jump in one frame. Measured on real
  clips: a correct box during a fast lunge jumped 0.284 x its own
  height, and the drifting box jumped 0.356. Too close to separate.
  Worse, the OTHER drift only jumped 0.194, less than the correct
  lunge. Any threshold either misses real drift or fires on real
  lunges.
* Comparing the tracker's motion against optical flow of the box
  contents. Correct tracking during a blurry lunge disagreed by 116px;
  an actual drift disagreed by 157px. Again overlapping.
* Raising CSRT's own psr_threshold so it gives up sooner. At 0.06 it
  changed nothing; at 0.10 it declared failure at frame 30, during a
  perfectly good lunge.

So this file deliberately contains only ONE check, the one that
actually holds up.

THE CHECK THAT WORKS: two boxes cannot be the same person
Two fencers cannot occupy the same space. If Fencer A's box and Fencer
B's box sit on top of each other, at least one of them is wrong. That is
geometry, not a tuned appearance threshold, so it does not care about
lighting, blur, or everybody wearing white.

On the clip above, both boxes converged onto a single fencer around
frame 175 and stayed there. This catches that.

It is deliberately allowed to overlap BRIEFLY, because in sabre a fleche
really does send fencers past each other at close quarters. Only a
sustained overlap is reported.

WHAT THIS DOES NOT CATCH
A single tracker drifting onto a background person while the other one
stays correct, like the referee case at frame 49, produces no
overlap and is NOT caught here. That failure still has to be spotted in
the live window. I would rather have one check that works than four that
produce false alarms people learn to ignore.
"""


def intersection_over_union(box1, box2) -> float:
    """
    How much two boxes overlap, from 0.0 (not at all) to 1.0 (identical).

    Called IoU: the shared area divided by the total area they cover
    between them.
    """
    ax, ay, aw, ah = box1
    bx, by, bw, bh = box2

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    if right <= left or bottom <= top:
        return 0.0

    overlap = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


class OverlapWatch:
    """
    Reports when the two fencers' boxes have been sitting on top of each
    other for long enough that it can't be explained by a close pass.
    """

    def __init__(self, iou_threshold=0.45, frames_before_warning=8):
        # 0.45 is a big overlap, roughly "these boxes are mostly the
        # same rectangle". Brushing past each other scores far lower.
        self.iou_threshold = iou_threshold
        # 8 frames is about a quarter of a second at 30fps. A fleche pass
        # clears in less; two trackers stuck on one person do not.
        self.frames_before_warning = frames_before_warning
        self.consecutive = 0
        self.warned = False

    def update(self, box_a, box_b, both_tracked: bool):
        """
        Returns (iou, is_warning, is_new_warning).

        is_new_warning is True only on the frame the warning first
        fires, so main.py can pause once instead of every frame.
        """
        if not both_tracked:
            self.consecutive = 0
            return 0.0, False, False

        iou = intersection_over_union(box_a, box_b)

        if iou >= self.iou_threshold:
            self.consecutive += 1
        else:
            self.consecutive = 0
            self.warned = False

        warning = self.consecutive >= self.frames_before_warning
        new_warning = warning and not self.warned
        if new_warning:
            self.warned = True

        return iou, warning, new_warning
