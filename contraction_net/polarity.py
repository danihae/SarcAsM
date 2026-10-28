"""Which side of a trace is rest, decided once per trace before the network reads it.

Rest is the level a trace dwells near. Once drift slower than 2 s (a running median) and
frame noise faster than 0.1 s (a boxcar) are removed, the residual of a beating trace sits
near zero at rest and makes excursions to one side only, so its Bowley skewness says which
side rest is on. :func:`polarity_score` turns that into one number per trace, positive when
rest is the high side, and :func:`decide_polarity` turns the number into a sign.

``predict_contractions`` multiplies the trace by that sign, so the network always reads a
rest-high trace and ``x`` and ``-x`` reach it as the same array. A trace too short, too
flat or too symmetric to decide gets sign 0 and no contractions, never the reading of the
wrong side. The constants are frozen in :class:`PolarityRule` and stored in the
checkpoint, next to the input convention they belong to.
"""

import math
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

__all__ = ['PolarityRule', 'PolarityDecision', 'polarity_score', 'decide_polarity']


@dataclass(frozen=True)
class PolarityRule:
    """Constants of the polarity decision, stored in a checkpoint as ``'polarity_rule'``.

    Parameters
    ----------
    version : int
        Rule version; only version 1 exists.
    reference_s : float
        Running-median window in s that sets the local rest level. Drift and steps slower
        than this are removed.
    smooth_s : float
        Boxcar window in s applied to the residual. Noise faster than this is removed.
    quantile : float
        Lower percentile ``q`` of the Bowley skewness, which uses ``q``, 50 and ``100 - q``.
    margin : float
        Scores with ``|s| <= margin`` abstain.
    min_frames, min_seconds : int, float
        Shorter traces abstain as ``'too_short'``: polarity needs about two beats.
    """

    version: int = 1
    reference_s: float = 2.0
    smooth_s: float = 0.1
    quantile: float = 10.0
    margin: float = 0.10
    min_frames: int = 32
    min_seconds: float = 2.0

    @classmethod
    def from_checkpoint(cls, stored):
        """Rule from a checkpoint's ``'polarity_rule'`` entry (a dict); None gives the defaults.

        Checkpoints trained before the rule existed store none, so they get version 1.
        Any other version, or a field version 1 does not have, is refused rather than
        guessed.
        """
        if stored is None:
            return cls()
        stored = dict(stored)
        if stored.get('version') != 1:
            raise ValueError(f"polarity_rule version {stored.get('version')!r} is not "
                             f"supported; this code knows version 1.")
        unknown = set(stored) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'polarity_rule has fields version 1 does not know: '
                             f'{sorted(unknown)}')
        return cls(**stored)


@dataclass(frozen=True)
class PolarityDecision:
    """Which side of one trace was taken as rest, and why.

    Attributes
    ----------
    sign : int
        +1: the trace is read as given (rest high); -1: it is negated first (rest low);
        0: undetermined, and the prediction is zeros.
    score : float
        :func:`polarity_score` of the trace, positive when rest is high. NaN when a guard
        ('flat', 'too_short') stopped the trace before it was scored.
    margin : float
        Abstention margin the score was compared with.
    reason : str
        'decided', 'ambiguous' (``|score| <= margin``), 'flat', 'too_short', 'declared', or
        'declared, data disagree' when the score points the other way by more than the
        margin. ``predict_contractions`` reports "rule not used: '<convention>' checkpoint"
        for checkpoints whose convention does not apply the rule.
    declared : int or None
        The polarity declared by the caller, if any.
    """

    sign: int
    score: float
    margin: float
    reason: str
    declared: Optional[int] = None


def _odd_frames(seconds, frametime, n):
    """Odd frame count nearest to ``seconds``, capped at the largest odd number <= ``n``.

    ``2 * floor(L / 2) + 1`` for ``L = seconds / frametime``: 2 s is 61 frames at 30 fps
    and 31 at 15 fps; 0.1 s is 3 frames at 30 fps and 1 (no smoothing) at 15 fps. An even
    ``L``, as 2 s is at any whole frame rate, lies halfway between two odd counts and takes
    the larger; the slack of 1e-9 keeps the last bit of ``1 / fps`` from deciding that tie
    (``2 / (1 / 93)`` is 185.99999999999997).
    """
    w = 2 * math.floor(seconds / frametime / 2 + 1e-9) + 1
    return max(1, min(w, n if n % 2 else n - 1))


def _local_skewness(x, frametime, rule):
    """Bowley skewness of ``x`` about its running-median rest level; ``g`` in the score."""
    n = x.size
    residual = x - median_filter(x, size=_odd_frames(rule.reference_s, frametime, n),
                                 mode='reflect')
    k = _odd_frames(rule.smooth_s, frametime, n)
    # a 1-frame boxcar means no smoothing; uniform_filter1d would still round its running
    # sum, so it is skipped rather than run
    if k > 1:
        residual = uniform_filter1d(residual, size=k, mode='reflect')
    lo, mid, hi = np.percentile(residual, [rule.quantile, 50.0, 100.0 - rule.quantile])
    span = hi - lo
    return float((2.0 * mid - lo - hi) / span) if span > 0 else 0.0


def _checked(x, frametime):
    """The trace as a float64 vector, which is exact for float16, float32 and integers."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if not np.isfinite(x).all():
        raise ValueError('Trace contains NaN or infinite values; fill gaps first.')
    if frametime is None or not np.isfinite(frametime) or frametime <= 0:
        raise ValueError(f'frametime must be a positive number of seconds, got {frametime!r}')
    return x, float(frametime)


def polarity_score(x, frametime, rule=None):
    """Polarity score of a trace: positive when rest is the high side.

    ``s = (g(x) - g(-x)) / 2``, where ``g`` is the Bowley skewness (10/50/90) of
    ``x - median_filter(x, 2 s)`` after a 0.1-s boxcar. Computed in float64 on the CPU.

    Parameters
    ----------
    x : ndarray
        1D trace, finite.
    frametime : float
        Time between frames in s; the windows are defined in seconds.
    rule : PolarityRule, optional
        Constants; default is version 1.

    Returns
    -------
    float
        The score, in ``[-1, 1]``. ``polarity_score(-x) == -polarity_score(x)`` exactly.
    """
    rule = PolarityRule() if rule is None else rule
    x, frametime = _checked(x, frametime)
    # g alone is not exactly odd: np.percentile interpolates P10 and P90 from opposite ends
    # of the sorted residual, so g(-x) and -g(x) can differ in the last bits. Their
    # difference is odd whatever g does, since IEEE subtraction gives a - b == -(b - a) and
    # halving is exact. This is what makes x and -x reach the network as the same array.
    return 0.5 * (_local_skewness(x, frametime, rule)
                  - _local_skewness(np.negative(x), frametime, rule))


def decide_polarity(x, frametime, polarity=None, margin=None, rule=None):
    """Decide which side of a trace is rest.

    Parameters
    ----------
    x : ndarray
        1D trace, finite.
    frametime : float
        Time between frames in s.
    polarity : {None, 1, -1}, optional
        None (default) decides from the trace. +1 or -1 declares the rest side (+1: high),
        for quantities whose sign physics fixes; the score is still computed and reported.
    margin : float, optional
        Abstain when ``|s| <= margin``. Default is the rule's (0.10). 0 decides every
        trace by the sign of its score, and abstains only when the score is exactly 0.
    rule : PolarityRule, optional
        Constants; default is version 1.

    Returns
    -------
    PolarityDecision

    Raises
    ------
    ValueError
        If the trace is not finite, or an argument is out of range.

    Notes
    -----
    Two guards run first, whatever is declared, and both are even in ``x``: traces shorter
    than ``min_frames`` or ``min_seconds`` are 'too_short', and traces whose 10th and 90th
    percentiles coincide on ``x`` or on ``-x`` (constants, zeros, a lone outlier or blips
    on zeros) are 'flat'. The comparisons with ``+-margin`` are symmetric, so ``-x`` always
    gets the opposite sign and the same reason, and ``|s| == margin`` abstains either way.
    """
    rule = PolarityRule() if rule is None else rule
    margin = rule.margin if margin is None else float(margin)
    if not margin >= 0:
        raise ValueError(f'margin must be >= 0, got {margin!r}')
    if polarity not in (None, 1, -1):
        raise ValueError(f'polarity must be None, 1 or -1, got {polarity!r}')
    declared = None if polarity is None else int(polarity)
    x, frametime = _checked(x, frametime)

    n = x.size
    # the relative slack keeps a trace of exactly min_seconds from being cut by the last
    # bit of n * frametime (98 frames at 49 fps are 1.9999999999999998 s)
    if n < rule.min_frames or n * frametime < rule.min_seconds * (1 - 1e-9):
        return PolarityDecision(0, math.nan, margin, 'too_short', declared)
    for v in (x, np.negative(x)):
        p10, p90 = np.percentile(v, [10.0, 90.0])
        if p10 == p90:
            return PolarityDecision(0, math.nan, margin, 'flat', declared)

    s = polarity_score(x, frametime, rule)
    if declared is not None:
        reason = 'declared, data disagree' if s * declared < -margin else 'declared'
        return PolarityDecision(declared, s, margin, reason, declared)
    sign = 1 if s > margin else (-1 if s < -margin else 0)
    return PolarityDecision(sign, s, margin, 'decided' if sign else 'ambiguous')
