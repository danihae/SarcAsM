"""From per-frame contraction probability to contraction intervals, one rule for every tool.

SarcAsM and CardioTrace call the same :func:`decode_contractions` with the same stored
threshold and confidence peak, so the same probabilities give the same contractions in
both. Only two inherited consumer settings differ: the minimum duration, and whether a
contraction cut off by the start or end of the recording is kept.

The decoder reads the probabilities and the frametime only, never the trace or its
polarity, so the contractions inherit the mirror invariance of
:func:`~contraction_net.prediction.predict_contractions`. An abstained trace is all zeros
and decodes to no contractions.
"""

import numpy as np
import torch

from .prediction import recommended_threshold

__all__ = ['DEFAULT_PEAK', 'decode_contractions', 'intervals_to_mask', 'recommended_decoding']

#: Confidence peak for checkpoints that store no ``'decoding'`` entry.
DEFAULT_PEAK = 0.85


def decode_contractions(prob, frametime, *, threshold, peak, min_duration_s, border_frames=0):
    """Contraction intervals of a contraction-state probability trace.

    Rules, in order:

    1. candidates are the maximal runs of frames where ``prob > threshold``;
    2. a run is kept only if its maximum reaches ``peak`` (hysteresis): a contraction must
       be confident somewhere, not only above the threshold;
    3. runs shorter than ``max(1, round(min_duration_s / frametime))`` frames are dropped;
    4. runs with any frame within ``border_frames`` of either end of the trace are dropped,
       as CardioTrace's ``clear_border_scipy`` does.

    No gap between runs is closed: closing erases the split the per-twitch label puts at
    the trough between overlapping twitches. A sustained contraction is one run, however
    long.

    Parameters
    ----------
    prob : ndarray
        1D contraction-state probability, channel 0 of
        :func:`~contraction_net.prediction.predict_contractions`.
    frametime : float
        Time between frames in s.
    threshold : float
        Probability a frame must exceed to be part of a run.
    peak : float
        Probability a run must reach at least once.
    min_duration_s : float
        Shortest contraction in s. Converted with Python's ``round()``, so the same seconds
        give the same frames in every tool.
    border_frames : int, optional
        Default 0 keeps contractions that touch either end; SarcAsM keeps them and flags
        them as incomplete cycles. CardioTrace drops those within 5 frames.

    Returns
    -------
    list of tuple of int
        One ``(start, stop)`` pair per contraction, half-open: frames ``start`` to
        ``stop - 1``.

    Raises
    ------
    ValueError
        If ``prob`` is not 1D, such as the whole output of ``predict_contractions``, whose
        channels would otherwise be decoded end to end.
    """
    prob = np.asarray(prob)
    if prob.ndim != 1:
        raise ValueError(f'prob must be 1D, got shape {prob.shape}: pass channel 0 of '
                         f'predict_contractions')
    if not frametime > 0:
        raise ValueError(f'frametime must be positive, got {frametime!r}')
    if border_frames < 0:
        raise ValueError(f'border_frames must be >= 0, got {border_frames!r}')
    n = prob.size
    min_frames = max(1, int(round(min_duration_s / frametime)))

    above = np.concatenate(([False], prob > threshold, [False]))
    edges = np.flatnonzero(above[1:] != above[:-1])
    intervals = []
    for start, stop in zip(edges[0::2].tolist(), edges[1::2].tolist()):
        if prob[start:stop].max() < peak:
            continue
        if stop - start < min_frames:
            continue
        if start < border_frames or stop > n - border_frames:
            continue
        intervals.append((start, stop))
    return intervals


def intervals_to_mask(intervals, n):
    """Boolean mask of ``n`` frames, True inside each half-open ``(start, stop)`` interval."""
    mask = np.zeros(int(n), dtype=bool)
    for start, stop in intervals:
        mask[start:stop] = True
    return mask


def recommended_decoding(model):
    """Threshold and confidence peak a checkpoint was tuned for.

    Both are properties of the model, stored in the checkpoint as ``'decoding'``.
    Checkpoints without the key fall back to
    :func:`~contraction_net.prediction.recommended_threshold` and :data:`DEFAULT_PEAK`.

    Parameters
    ----------
    model : str or torch.nn.Module
        Checkpoint path or module.

    Returns
    -------
    dict
        ``{'threshold': float, 'peak': float}``, the keyword arguments of
        :func:`decode_contractions` that belong to the model.
    """
    if isinstance(model, torch.nn.Module):
        stored = getattr(model, 'decoding', None)
    else:
        stored = torch.load(model, map_location='cpu', weights_only=True).get('decoding')
    stored = dict(stored or {})
    threshold = stored.get('threshold')
    if threshold is None:
        threshold = recommended_threshold(model)
    return {'threshold': float(threshold), 'peak': float(stored.get('peak', DEFAULT_PEAK))}
