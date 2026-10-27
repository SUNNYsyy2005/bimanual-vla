"""Shared exception identities for imported replay code and its -m worker."""


class ReplayBlocked(RuntimeError):
    pass


class ReplayFeedbackPending(ReplayBlocked):
    """No complete fresh feedback yet; only initial connection may wait."""


class ReplayCancelled(Exception):
    pass
