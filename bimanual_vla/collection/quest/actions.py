"""Collection shortcuts and bounded joint-space reset interpolation."""
import math

import numpy as np


class CollectionButtons:
    def __init__(self):
        self.clear()

    def clear(self):
        self.sequence = None
        self.armed = False
        self.x = False
        self.a_since = None
        self.a_fired = False

    def update(self, sample):
        if sample is None:
            self.clear()
            return []
        if sample.source_time == self.sequence:
            return []
        self.sequence = sample.source_time
        b = sample.buttons
        x, a = bool(b.get('X')), bool(b.get('A'))
        if not self.armed:
            self.armed = not x and not a
            return []
        events = ['toggle_record'] if x and not self.x else []
        self.x = x
        if not a or b.get('Y') or b.get('B'):
            self.a_since, self.a_fired = None, False
        elif self.a_since is None:
            self.a_since = sample.received_monotonic
        elif not self.a_fired and sample.received_monotonic - self.a_since >= 1.:
            self.a_fired = True
            events.append('reset_initial')
        return events


class JointResetPlan:
    """Quintic rest-to-rest path. This is only used for explicit resets."""
    def __init__(self, start, target, *, velocity=.10, acceleration=.30):
        self.start = np.asarray(start, dtype=float).copy()
        self.target = np.asarray(target, dtype=float).copy()
        if (self.start.shape != self.target.shape or self.start.ndim != 1
                or not np.isfinite(self.start).all() or not np.isfinite(self.target).all()):
            raise ValueError('invalid reset joint vectors')
        distance = float(np.max(np.abs(self.target - self.start)))
        self.duration = max(1., 1.875 * distance / velocity,
                            math.sqrt(5.774 * distance / acceleration))

    def at(self, elapsed):
        u = np.clip(elapsed / self.duration, 0., 1.)
        blend = u**3 * (10 + u * (-15 + 6*u))
        return self.start + blend * (self.target - self.start)
