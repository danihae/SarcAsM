"""
Benchmark harness for ContractionNet.

Measures the two things a contraction detector is easiest to get wrong: what fraction of a
recording is spent contracting ("duty"), and how coarsely a contraction is sampled. Both are
properties of the *recording*, not of the contraction, and a detector that quietly depends on
either will look fine on typical data and fail on the rest. Three complementary probes:

- :func:`duty_duration_grid` -- a controlled sweep over (event duration x duty cycle) on
  idealised traces, which localises the cliff precisely;
- :func:`score_traces` -- realistic traces from :mod:`contraction_net.simulation`, scored
  per duty bin and per regime. Run this on :func:`~contraction_net.simulation.make_stress_set`,
  whose seed and parameter ranges are disjoint from the training generator, so a good score
  means generalisation rather than a memorised simulator;
- :func:`real_data_report` -- per-row duty and cycle counts on the packaged high-speed
  recordings, plus :func:`determinism_report` and :func:`offset_invariance_report`, which
  catch inference bugs that synthetic scores cannot see.

Run against one or more checkpoints::

    python -m contraction_net.benchmark --model sarcasm/models/model_ContractionNet.pt
    python -m contraction_net.benchmark --model old.pt --model new.pt
"""

import argparse
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import label as _label

from .prediction import recommended_threshold
from .simulation import (REST_TOL, Trace, make_stress_set, twitch_waveform,
                         _event_frames, _label_from_clean)

__all__ = [
    'PredictFn',
    'model_predictor',
    'model_predictor_full',
    'boundary_report',
    'polarity_invariance_report',
    'polarity_cost_probe',
    'cross_grouping_consistency',
    'corpus_report',
    'grid_trace',
    'oracle_amplitude_detector',
    'reference_ceiling',
    'postprocess',
    'duty_duration_grid',
    'score_traces',
    'sampling_sweep',
    'real_data_report',
    'determinism_report',
    'offset_invariance_report',
    'DEFAULT_DURATIONS',
    'DEFAULT_DUTIES',
]

#: A predictor takes a 1D trace and its frametime and returns a per-frame contraction
#: probability of the same length. Frametime is passed even though the legacy model ignores
#: it, so frame-rate-aware models can be scored by the same harness.
PredictFn = Callable[[np.ndarray, float], np.ndarray]

DEFAULT_DURATIONS = (20, 40, 80, 150, 250)
DEFAULT_DUTIES = (0.20, 0.40, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)

_REAL_DATA = 'test_data/high_speed_single_ACTN2-citrine_CM'


def model_predictor(model_path: str, channel: int = 0) -> PredictFn:
    """
    Wrap a checkpoint into a :data:`PredictFn`.

    Parameters
    ----------
    model_path : str
        Path to a ContractionNet ``.pt`` checkpoint.
    channel : int, optional
        Output channel to return. 0 is the contraction state, 1 and 2 the onset and offset
        heads. Default is 0.

    Returns
    -------
    PredictFn
        Callable returning the per-frame probability of the chosen channel.
    """
    from .prediction import predict_contractions

    def _predict(signal: np.ndarray, frametime: float) -> np.ndarray:
        return predict_contractions(np.asarray(signal, dtype=float), model_path)[channel]

    return _predict


def model_predictor_full(model_path: str):
    """Wrap a checkpoint into a callable returning **all** output channels."""
    from .prediction import predict_contractions

    def _predict(signal: np.ndarray, frametime: float) -> np.ndarray:
        return predict_contractions(np.asarray(signal, dtype=float), model_path)

    return _predict


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def postprocess(mask: np.ndarray, frametime: float = None, min_frames: int = 3,
                merge_frames: int = 2, contr_time_min: float = None,
                merge_time_max: float = None) -> np.ndarray:
    """
    Apply the morphological cleanup that SarcAsM runs on every prediction.

    :func:`sarcasm.analysis.contraction_analysis.detect_contractions` closes short gaps and
    removes short intervals before anything downstream sees the mask. Scoring raw
    thresholded probabilities would measure something users never get, and would exaggerate
    fragmentation on noisy traces.

    The cleanup is expressed in **frames** by default rather than in seconds. SarcAsM's own
    defaults (0.2 s minimum duration) encode cardiomyocyte timescales, and applying them
    here would delete every contraction in a coarsely sampled recording -- scoring as a
    model miss what is really a filter the benchmark imposed. The detector has to work on
    time-series whose sampling resolves a contraction in only a handful of frames, so the
    benchmark filters at the resolvability limit instead.

    Parameters
    ----------
    mask : np.ndarray
        Binary contraction state.
    frametime : float, optional
        Time between frames in s. Only needed with the seconds-based parameters.
    min_frames : int, optional
        Minimal contraction duration in frames. Default is 3.
    merge_frames : int, optional
        Maximal gap in frames between merged contractions. Default is 2.
    contr_time_min, merge_time_max : float, optional
        Seconds-based equivalents, for mirroring a specific pipeline configuration. When
        given they override the frame-based values and require ``frametime``.

    Returns
    -------
    np.ndarray
        Cleaned boolean mask.
    """
    from scipy.ndimage import binary_closing, binary_opening
    if contr_time_min is not None or merge_time_max is not None:
        if frametime is None:
            raise ValueError('frametime is required with the seconds-based parameters.')
        if contr_time_min is not None:
            min_frames = max(1, int(contr_time_min / frametime))
        if merge_time_max is not None:
            merge_frames = max(1, int(merge_time_max / frametime))
    closing = np.ones(max(1, int(merge_frames)))
    opening = np.ones(max(1, int(min_frames)))
    return binary_opening(binary_closing(np.asarray(mask, dtype=bool), structure=closing),
                          structure=opening)


def oracle_amplitude_detector(signal: np.ndarray, amplitude: float,
                              enter: float = 0.20, exit_: float = 0.06) -> np.ndarray:
    """
    Reference detector that is told the true contraction amplitude.

    This exists to bound the benchmark, not to compete with the network. A synthetic set
    can always be made unsolvable by piling on noise, and then a low model score says
    nothing; reporting this alongside the model lets the numbers be read against an
    achievable ceiling instead of against 1.0.

    It is deliberately **oracle-assisted**: it receives the amplitude used to build the
    trace, so it never has to solve blind amplitude estimation. That is on purpose. The
    question this harness asks is whether the *temporal segmentation* is recoverable from a
    noisy trace, and mixing in a hard, separate estimation problem would understate the
    ceiling and make the bound useless. It never sees the labels.

    The resting level is a rolling high percentile of the smoothed trace: rest is the high
    side of a length trace, so a percentile tracks it through a sustained contraction, and
    unlike a rolling maximum it is not biased several sigma upward on a flat trace.

    Parameters
    ----------
    signal : np.ndarray
        1D sarcomere length trace.
    amplitude : float
        True contraction amplitude in µm.
    enter, exit_ : float, optional
        Hysteresis thresholds as a fraction of ``amplitude``.

    Returns
    -------
    np.ndarray
        Per-frame pseudo-probability in ``{0, 1}``.
    """
    from scipy.ndimage import label as _lab, percentile_filter, uniform_filter1d
    x = np.asarray(signal, dtype=float)
    if x.size < 16 or amplitude <= 0:
        return np.zeros_like(x)

    sm = uniform_filter1d(x, size=5, mode='nearest')
    win = int(np.clip(x.size // 2, 25, x.size))
    rest = percentile_filter(sm, 90, size=win, mode='nearest')
    depth = (rest - sm) / amplitude

    strong, weak = depth > enter, depth > exit_
    lab, n = _lab(weak)
    keep = np.zeros_like(weak)
    for i in range(1, n + 1):
        m = lab == i
        if strong[m].any():
            keep |= m
    return keep.astype(float)


def reference_ceiling(traces: Sequence[Trace], threshold: float = 0.3) -> Dict[str, object]:
    """
    Score :func:`oracle_amplitude_detector` on a set of traces.

    Returns the same structure as :func:`score_traces`, plus ``noiseless`` -- the same
    detector run on ``Trace.clean``. The two together separate the two reasons a ceiling
    can sit below 1.0: ``noiseless`` is what this reference method costs by itself (its
    rolling-percentile baseline and fixed thresholds never reproduce the label rule
    exactly), and the gap down to ``overall`` is what the noise costs. Model numbers should
    be read against ``overall``.
    """
    ious, duties, regimes, cnt_err, quiescent_fp, clean_ious, fpe = [], [], [], [], [], [], []
    for tr in traces:
        amp = tr.meta.get('amplitude', 0.0)
        mask = postprocess(_binary(oracle_amplitude_detector(tr.signal, amp), 0.5),
                           tr.frametime).astype(float)
        if not np.asarray(tr.label, dtype=bool).any():
            quiescent_fp.append(float((mask > 0.5).mean()))
            continue
        clean_mask = postprocess(_binary(oracle_amplitude_detector(tr.clean, amp), 0.5),
                                 tr.frametime).astype(float)
        clean_ious.append(iou(clean_mask, tr.label, 0.5))
        ious.append(iou(mask, tr.label, 0.5))
        cnt_err.append(abs(event_count_error(mask, tr.label, 0.5)))
        duties.append(tr.duty)
        regimes.append(tr.regime)
        fpe.append(_frames_per_event(tr.label))

    ious, duties, fpe = np.asarray(ious), np.asarray(duties), np.asarray(fpe, dtype=float)
    by_duty = {}
    for lo in np.arange(0.0, 1.0, 0.1):
        m = (duties >= lo) & (duties < lo + 0.1)
        if m.any():
            by_duty[round(float(lo), 1)] = (int(m.sum()), float(ious[m].mean()))
    by_regime = {}
    for reg in sorted(set(regimes)):
        m = np.array([r == reg for r in regimes])
        by_regime[reg] = (int(m.sum()), float(ious[m].mean()),
                          float(np.asarray(cnt_err, dtype=float)[m].mean()))
    by_sampling = {}
    for lo, hi in ((3, 6), (6, 12), (12, 25), (25, 60), (60, 150), (150, 10 ** 6)):
        m = (fpe >= lo) & (fpe < hi)
        if m.any():
            by_sampling[(lo, hi)] = (int(m.sum()), float(ious[m].mean()))

    return {'overall': float(ious.mean()) if ious.size else np.nan,
            'noiseless': float(np.mean(clean_ious)) if clean_ious else np.nan,
            'by_duty': by_duty, 'by_regime': by_regime, 'by_sampling': by_sampling,
            'quiescent_fp': float(np.mean(quiescent_fp)) if quiescent_fp else np.nan,
            'n_quiescent': len(quiescent_fp)}


def _binary(prob: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(prob) > threshold


def iou(prob: np.ndarray, truth: np.ndarray, threshold: float = 0.3) -> float:
    """Intersection over union between the thresholded prediction and the ground truth."""
    pred, truth = _binary(prob, threshold), np.asarray(truth, dtype=bool)
    union = int((pred | truth).sum())
    if union == 0:
        return 1.0  # both empty: a correctly detected quiescent trace
    return float((pred & truth).sum() / union)


def recall_precision(prob: np.ndarray, truth: np.ndarray, threshold: float = 0.3):
    """Per-frame recall and precision. NaN where the denominator is empty."""
    pred, truth = _binary(prob, threshold), np.asarray(truth, dtype=bool)
    tp = float((pred & truth).sum())
    rec = tp / truth.sum() if truth.any() else np.nan
    pre = tp / pred.sum() if pred.any() else np.nan
    return rec, pre


def event_count_error(prob: np.ndarray, truth: np.ndarray, threshold: float = 0.3) -> int:
    """Signed error in the number of contraction intervals.

    Beating rate is derived from cycle onsets, so miscounting events is the error that
    propagates furthest downstream -- a trace can score a decent IoU while reporting the
    wrong number of beats.
    """
    return int(_label(_binary(prob, threshold))[1]) - int(_label(np.asarray(truth, bool))[1])


# --------------------------------------------------------------------------------------
# controlled synthetic probes
# --------------------------------------------------------------------------------------

def grid_trace(duration: int, duty: float, n_frames: int = 1000, seed: int = 0,
               amplitude: float = 0.15, baseline: float = 1.75, noise: float = 0.015):
    """
    Idealised periodic trace with an exactly prescribed event duration and duty cycle.

    Deliberately cleaner and more regular than
    :func:`~contraction_net.simulation.simulate_trace`: the point of the grid is to vary
    exactly two quantities and hold everything else fixed, so a drop in score can only be
    attributed to duration or duty.

    Parameters
    ----------
    duration : int
        Contraction duration in frames.
    duty : float
        Target fraction of frames spent contracting.
    n_frames : int, optional
        Trace length. Default is 1000.
    seed : int, optional
        Random seed. Default is 0.
    amplitude : float, optional
        Contraction amplitude in µm. Default is 0.15.
    baseline : float, optional
        Resting sarcomere length in µm. Default is 1.75.
    noise : float, optional
        Gaussian noise standard deviation in µm. Default is 0.015.

    Returns
    -------
    tuple of np.ndarray
        ``(signal, label)``.

    Notes
    -----
    The label is derived from the noiseless signal via
    :data:`~contraction_net.simulation.REST_TOL`, exactly as
    :func:`~contraction_net.simulation.simulate_trace` does. Labelling the full scheduled
    duration instead would mark the shallow tail of the relaxation as contracting even
    though it sits at rest, capping the achievable IoU near 0.7 and hiding whatever the
    model actually does.
    """
    rng = np.random.default_rng(seed)
    # mostly plateau with a fast relaxation, so the labelled interval is essentially the
    # scheduled one and `duration` means what it says
    fracs = (0.15, 0.60, 6.0)
    shape = twitch_waveform(duration, *fracs)
    labelled = max(1, int((shape > REST_TOL).sum()))
    gap = max(1, int(round(labelled * (1 - duty) / max(duty, 1e-6))))
    signal = np.full(n_frames, baseline)
    start = 5
    while start + duration <= n_frames:
        signal[start:start + duration] -= amplitude * shape
        start += duration + gap
    truth = _label_from_clean(signal, baseline, amplitude)
    return signal + rng.normal(0, noise, n_frames), truth


def duty_duration_grid(predict: PredictFn, durations: Sequence[int] = DEFAULT_DURATIONS,
                       duties: Sequence[float] = DEFAULT_DUTIES, n_frames: int = 1000,
                       frametime: float = 0.0164, n_seeds: int = 3,
                       threshold: float = 0.3) -> Dict[str, np.ndarray]:
    """
    Sweep event duration against duty cycle and report mean IoU in each cell.

    Parameters
    ----------
    predict : PredictFn
        Model wrapper, e.g. from :func:`model_predictor`.
    durations : sequence of int, optional
        Contraction durations in frames.
    duties : sequence of float, optional
        Target duty cycles.
    n_frames : int, optional
        Trace length. Default is 1000.
    frametime : float, optional
        Frametime handed to ``predict``. Default is 0.0164 s (the packaged 61 fps data).
    n_seeds : int, optional
        Repeats per cell. Default is 3.
    threshold : float, optional
        Binarisation threshold. Default is 0.3.

    Returns
    -------
    dict
        ``{'durations', 'duties', 'iou'}`` with ``iou`` of shape
        ``(len(durations), len(duties))``.
    """
    out = np.full((len(durations), len(duties)), np.nan)
    for i, dur in enumerate(durations):
        for j, duty in enumerate(duties):
            vals = []
            for s in range(n_seeds):
                sig, truth = grid_trace(dur, duty, n_frames=n_frames, seed=s)
                if not truth.any():
                    continue
                mask = postprocess(_binary(predict(sig, frametime), threshold), frametime)
                vals.append(iou(mask.astype(float), truth, 0.5))
            if vals:
                out[i, j] = float(np.mean(vals))
    return {'durations': np.asarray(durations), 'duties': np.asarray(duties), 'iou': out}


def _frames_per_event(label: np.ndarray) -> float:
    """Median contraction length in frames -- the sampling-quality axis."""
    lab, n = _label(np.asarray(label, dtype=bool))
    if n == 0:
        return np.nan
    return float(np.median([int((lab == i).sum()) for i in range(1, n + 1)]))


def sampling_sweep(predict: PredictFn, duty: float = 0.45, n_frames: int = 1000,
                   frames_per_event: Sequence[int] = (3, 4, 6, 10, 20, 50, 150, 400),
                   frametime: float = 0.0164, n_seeds: int = 3,
                   threshold: float = 0.3) -> Dict[str, np.ndarray]:
    """
    Sweep how many frames resolve one contraction, holding duty fixed.

    The detector is meant to generalise beyond high-speed sarcomere recordings to
    time-series with much poorer temporal sampling, where a whole contraction may be
    covered by three or four samples. This isolates that axis: the waveform and duty are
    identical in every cell, only the sampling changes.

    Returns
    -------
    dict
        ``{'frames_per_event', 'iou'}``.
    """
    out = np.full(len(frames_per_event), np.nan)
    for i, fpe in enumerate(frames_per_event):
        vals = []
        for s in range(n_seeds):
            sig, truth = grid_trace(int(fpe), duty, n_frames=n_frames, seed=s)
            if not truth.any():
                continue
            mask = postprocess(_binary(predict(sig, frametime), threshold), min_frames=2)
            vals.append(iou(mask.astype(float), truth, 0.5))
        if vals:
            out[i] = float(np.mean(vals))
    return {'frames_per_event': np.asarray(frames_per_event), 'iou': out}


def score_traces(predict: PredictFn, traces: Sequence[Trace], threshold: float = 0.3
                 ) -> Dict[str, object]:
    """
    Score realistic simulated traces, broken down by duty bin and by regime.

    Parameters
    ----------
    predict : PredictFn
        Model wrapper.
    traces : sequence of Trace
        Traces from :mod:`contraction_net.simulation`.
    threshold : float, optional
        Binarisation threshold. Default is 0.3.

    Returns
    -------
    dict
        ``'overall'`` mean IoU over traces that contain contractions, ``'by_duty'`` mapping
        decile -> (n, mean IoU), ``'by_regime'`` mapping regime -> (n, mean IoU, mean
        |event-count error|), and ``'quiescent_fp'`` -- the mean fraction of frames wrongly
        flagged on traces that contain no contraction at all.

    Notes
    -----
    Quiescent traces are scored separately. IoU is undefined-to-brutal when the ground
    truth is empty (a single false-positive frame scores 0), so folding them into the mean
    would report a detector's behaviour on flat traces as if it were its segmentation
    accuracy. False-positive rate answers that question directly.
    """
    ious, duties, regimes, cnt_err, fpe = [], [], [], [], []
    quiescent_fp = []
    for tr in traces:
        prob = predict(tr.signal, tr.frametime)
        mask = postprocess(_binary(prob, threshold), tr.frametime).astype(float)
        if not np.asarray(tr.label, dtype=bool).any():
            quiescent_fp.append(float((mask > 0.5).mean()))
            continue
        ious.append(iou(mask, tr.label, 0.5))
        cnt_err.append(abs(event_count_error(mask, tr.label, 0.5)))
        duties.append(tr.duty)
        regimes.append(tr.regime)
        fpe.append(_frames_per_event(tr.label))
    ious = np.asarray(ious)
    duties = np.asarray(duties)
    cnt_err = np.asarray(cnt_err, dtype=float)

    by_duty = {}
    for lo in np.arange(0.0, 1.0, 0.1):
        m = (duties >= lo) & (duties < lo + 0.1)
        if m.any():
            by_duty[round(float(lo), 1)] = (int(m.sum()), float(ious[m].mean()))

    by_regime = {}
    for reg in sorted(set(regimes)):
        m = np.array([r == reg for r in regimes])
        by_regime[reg] = (int(m.sum()), float(ious[m].mean()), float(cnt_err[m].mean()))

    # Sampling quality: how many frames resolve one contraction. Bucketed in log space
    # because the interesting contrast is 3-vs-30 frames, not 300-vs-330.
    fpe = np.asarray(fpe, dtype=float)
    by_sampling = {}
    for lo, hi in ((3, 6), (6, 12), (12, 25), (25, 60), (60, 150), (150, 10 ** 6)):
        m = (fpe >= lo) & (fpe < hi)
        if m.any():
            by_sampling[(lo, hi)] = (int(m.sum()), float(ious[m].mean()))

    return {'overall': float(ious.mean()) if ious.size else np.nan,
            'by_duty': by_duty, 'by_regime': by_regime, 'by_sampling': by_sampling,
            'quiescent_fp': float(np.mean(quiescent_fp)) if quiescent_fp else np.nan,
            'n_quiescent': len(quiescent_fp),
            'iou': ious, 'duty': duties}


# --------------------------------------------------------------------------------------
# real-data and inference-hygiene probes
# --------------------------------------------------------------------------------------

def _load_real_rows(repo_root: str, name: str) -> Optional[np.ndarray]:
    """Per-myofibril sarcomere length time-series from a packaged OME-Zarr store."""
    import zarr
    path = os.path.join(repo_root, _REAL_DATA, f'{name}.ome.zarr')
    if not os.path.isdir(path):
        return None
    try:
        store = zarr.open(path, mode='r')
        return np.asarray(store['sarcasm/structure/mband/slen_timeseries'])
    except (KeyError, FileNotFoundError):
        return None


def _fill_nans(trace: np.ndarray) -> np.ndarray:
    """Linear gap fill, matching what the inference path does before prediction."""
    trace = np.asarray(trace, dtype=float)
    mask = np.isnan(trace)
    if not mask.any():
        return trace
    if mask.all():
        return np.zeros_like(trace)
    idx = np.arange(trace.size)
    out = trace.copy()
    out[mask] = np.interp(idx[mask], idx[~mask], trace[~mask])
    return out


def real_data_report(predict: PredictFn, repo_root: str = '.',
                     names: Sequence[str] = ('10kPa', '20kPa', '30kPa'),
                     frametime: float = 0.0164, threshold: float = 0.3
                     ) -> Dict[str, Dict[str, float]]:
    """
    Per-row duty and cycle counts on the packaged high-speed recordings.

    There is no ground truth here, so this reports distribution statistics rather than a
    score. The diagnostic signal is saturation: if nearly every row comes back at duty
    ~0.7 the model is pinned against its cliff rather than measuring the cells.

    Parameters
    ----------
    predict : PredictFn
        Model wrapper.
    repo_root : str, optional
        Repository root containing ``test_data/``. Default is ``'.'``.
    names : sequence of str, optional
        Recording names to load. Default is the three stiffness examples.
    frametime : float, optional
        Frametime of the recordings. Default is 0.0164 s.
    threshold : float, optional
        Binarisation threshold. Default is 0.3.

    Returns
    -------
    dict
        Per recording: ``n_rows``, ``duty_mean``, ``duty_max``, ``frac_rows_above_0.70``,
        ``n_contr_median``, ``longest_event_max``.
    """
    report = {}
    for name in names:
        rows = _load_real_rows(repo_root, name)
        if rows is None:
            continue
        duties, counts, longest = [], [], []
        for row in rows:
            prob = predict(_fill_nans(row), frametime)
            b = _binary(prob, threshold)
            duties.append(float(b.mean()))
            lab, n = _label(b)
            counts.append(n)
            longest.append(max((int((lab == i).sum()) for i in range(1, n + 1)), default=0))
        duties = np.asarray(duties)
        report[name] = {
            'n_rows': len(duties),
            'duty_mean': float(duties.mean()),
            'duty_max': float(duties.max()),
            'frac_rows_above_0.70': float((duties > 0.70).mean()),
            'n_contr_median': float(np.median(counts)),
            'longest_event_max': int(max(longest)),
        }
    return report


def determinism_report(predict: PredictFn, signal: np.ndarray, frametime: float = 0.0164,
                       n_repeats: int = 6, threshold: float = 0.3) -> Dict[str, float]:
    """
    Check that repeated predictions on one trace agree.

    A model left in training mode keeps dropout active at inference, which makes every call
    an unintended random ensemble member and makes cycle counts wobble between runs.

    Returns
    -------
    dict
        ``max_prob_spread``, ``frac_frames_unstable``, ``n_contr_values``.
    """
    probs = np.stack([predict(signal, frametime) for _ in range(n_repeats)])
    b = probs > threshold
    unstable = b.any(0) & ~b.all(0)
    return {
        'max_prob_spread': float(probs.max(0).max() - probs.min(0).min()) if probs.size else 0.0,
        'max_pointwise_spread': float((probs.max(0) - probs.min(0)).max()),
        'frac_frames_unstable': float(unstable.mean()),
        'n_contr_values': sorted({int(_label(x)[1]) for x in b}),
    }


def offset_invariance_report(predict: PredictFn, signal: np.ndarray,
                             frametime: float = 0.0164, offsets: Sequence[float] = (5.0, -1.0),
                             threshold: float = 0.3) -> Dict[str, float]:
    """
    Check that adding a constant to the input does not change the prediction.

    A constant offset carries no information about contraction, so the answer must not move.
    It does when the input is left uncentred and the convolutions zero-pad: the padding step
    scales with the offset and leaks into the normalisation statistics of the whole trace.

    Returns
    -------
    dict
        Fraction of frames whose binary label changes, per offset, plus the centred input.
    """
    base = _binary(predict(signal, frametime), threshold)
    out = {}
    for off in offsets:
        shifted = _binary(predict(np.asarray(signal) + off, frametime), threshold)
        out[f'offset_{off:+g}'] = float((base != shifted).mean())
    centred = _binary(predict(np.asarray(signal) - np.median(signal), frametime), threshold)
    out['centred'] = float((base != centred).mean())
    return out


# --------------------------------------------------------------------------------------
# polarity, boundaries and distilled-corpus probes
# --------------------------------------------------------------------------------------

def boundary_report(predict_full, traces: Sequence[Trace], tol_frames: int = 3,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Score the onset and offset heads, which nothing else in this harness looks at.

    Beating rate comes from onsets and cycle duration from onset/offset pairs, so a model
    can score well on the mask and still be the wrong choice downstream.

    Parameters
    ----------
    predict_full : callable
        Returns all output channels, e.g. :func:`model_predictor_full`.
    traces : sequence of Trace
    tol_frames : int, optional
        Frames within which a predicted transition counts as matched. Default is 3.
    threshold : float, optional
        Threshold on the boundary channels. Default is 0.5.

    Returns
    -------
    dict
        ``onset_f1``, ``offset_f1`` and ``onset_median_error`` in frames.
    """
    stats = {'onset': [0, 0, 0, []], 'offset': [0, 0, 0, []]}   # tp, fp, fn, errors
    for trace in traces:
        out = predict_full(trace.signal, trace.frametime)
        if out.shape[0] < 3:
            continue
        edges = np.diff(np.asarray(trace.label, np.int8))
        truth = {'onset': np.flatnonzero(edges > 0) + 1,
                 'offset': np.flatnonzero(edges < 0) + 1}
        for k, channel in (('onset', 1), ('offset', 2)):
            peaks = _peaks(out[channel], threshold)
            matched, used = 0, set()
            for position in truth[k]:
                near = [p for p in peaks if abs(p - position) <= tol_frames and p not in used]
                if near:
                    best = min(near, key=lambda p: abs(p - position))
                    used.add(best)
                    matched += 1
                    stats[k][3].append(abs(best - position))
            stats[k][0] += matched
            stats[k][1] += len(peaks) - len(used)
            stats[k][2] += len(truth[k]) - matched
    out = {}
    for k, (tp, fp, fn, errors) in stats.items():
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
              else np.nan)
        out[f'{k}_f1'] = float(f1)
        out[f'{k}_median_error'] = float(np.median(errors)) if errors else np.nan
    return out


def _peaks(prob: np.ndarray, threshold: float) -> List[int]:
    """One position per supra-threshold run: the boundary heads emit short tents."""
    mask = np.asarray(prob) > threshold
    lab, n = _label(mask)
    return [int(np.argmax(np.asarray(prob) * (lab == i))) for i in range(1, n + 1)]


def polarity_invariance_report(predict: PredictFn, traces: Sequence[Trace],
                               threshold: float = 0.5) -> Dict[str, float]:
    """How much a model's answer changes when its input is negated.

    Z-band position rows carry either sign depending on which side of the contraction node
    a band sits, and :meth:`sarcasm.motion.Motion.predict_contractions` feeds both to the
    network, so a polarity-dependent model is answering two different questions on one cell.

    Returns
    -------
    dict
        ``iou_upright`` and ``iou_flipped`` against the ground truth, and ``disagreement``
        -- the fraction of frames whose binary answer differs between the two. An invariant
        model gives ~0 for the last.
    """
    upright, flipped, disagree = [], [], []
    for trace in traces:
        baseline = float(np.median(trace.signal))
        a = predict(trace.signal, trace.frametime)
        b = predict(2 * baseline - trace.signal, trace.frametime)
        upright.append(iou(a, trace.label, threshold))
        flipped.append(iou(b, trace.label, threshold))
        disagree.append(float((_binary(a, threshold) != _binary(b, threshold)).mean()))
    return {'iou_upright': float(np.mean(upright)),
            'iou_flipped': float(np.mean(flipped)),
            'disagreement': float(np.mean(disagree)),
            'n': len(traces)}


def polarity_cost_probe(predict: PredictFn, threshold: float = 0.5,
                        duties: Sequence[float] = (0.20, 0.40, 0.60, 0.75, 0.85, 0.95),
                        duration: int = 80, n_frames: int = 1200) -> Dict[str, object]:
    """What polarity invariance costs, resolved by duty cycle.

    A polarity-invariant model cannot use "rest is the high side" and has to read rest off
    the waveform asymmetry instead -- fast rise, plateau, slow relaxation. That cue is
    weakest at high duty, where a single sustained event offers one transition, so this is
    where invariance should cost something if it costs anything.

    The symmetric-waveform control is the floor: with a linear relaxation and rise equal to
    relax, the two polarities are genuinely indistinguishable and **any** model must be at
    chance. Reporting it separates "the model lost the cue" from "the cue is not there".

    Returns
    -------
    dict
        ``duties``, ``iou_upright``, ``iou_flipped`` and ``gap`` per duty, plus
        ``symmetric_control`` holding the same for a waveform carrying no asymmetry.
    """
    frametime = 0.0164
    up, down, sym_up, sym_down = [], [], [], []
    for duty in duties:
        for signal, label, into_up, into_down in (
                grid_trace(duration, duty, n_frames=n_frames, seed=7) + (up, down),
                _symmetric_waveform_trace(duration, duty, n_frames) + (sym_up, sym_down)):
            base = float(np.median(signal))
            into_up.append(iou(predict(signal, frametime), label, threshold))
            into_down.append(iou(predict(2 * base - signal, frametime), label, threshold))
    up, down = np.asarray(up), np.asarray(down)
    return {'duties': np.asarray(duties), 'iou_upright': up, 'iou_flipped': down,
            'gap': up - down,
            'symmetric_control': {'iou_upright': np.asarray(sym_up),
                                  'iou_flipped': np.asarray(sym_down)}}


def _symmetric_waveform_trace(duration: int, duty: float, n_frames: int):
    """``(signal, label)`` whose twitch is time-symmetric, so polarity says nothing."""
    period = max(int(round(duration / max(duty, 1e-6))), duration + 2)
    amplitude, baseline = 0.15, 1.75
    clean = np.full(n_frames, baseline)
    half = max(duration // 2, 1)
    shape = np.concatenate([np.linspace(0, 1, half), np.linspace(1, 0, duration - half)])
    for start in range(0, n_frames - duration, period):
        clean[start:start + duration] -= amplitude * shape
    label = _label_from_clean(clean, baseline, amplitude)
    rng = np.random.default_rng(11)
    return clean + rng.normal(0, 0.015, n_frames), label


def cross_grouping_consistency(predict: PredictFn, store_paths: Sequence[str],
                               threshold: float = 0.5,
                               kinds: Sequence[str] = ('pool', 'mband', 'myofibril')
                               ) -> Dict[str, object]:
    """Agreement between a model's answers on the different groupings of one cell.

    Uses no labels at all: pool, M-band and myofibril signals describe the same cell, so a
    model that answers them consistently is measuring the cell rather than the aggregation.

    Returns
    -------
    dict
        Median pairwise IoU and duty correlation per kind pair, over the given stores.
    """
    from sarcasm.io.results_store import Results

    pairs, duties = {}, {}
    for path in store_paths:
        try:
            res = Results(path if path.rstrip('/').endswith('sarcasm')
                          else os.path.join(path, 'sarcasm'))
            frametime = float(res.metadata['frametime'])
            masks = {}
            for kind in kinds:
                rows = np.asarray(getattr(res.structure, kind).slen_timeseries, float)
                if rows.size == 0:
                    continue
                preds = [_binary(predict(_fill_nans(row), frametime), threshold)
                         for row in rows]
                masks[kind] = preds
        except Exception:
            continue
        for i, a in enumerate(kinds):
            for b in kinds[i + 1:]:
                if a not in masks or b not in masks:
                    continue
                score = np.median([[iou(x.astype(float), y, 0.5) for y in masks[b]]
                                   for x in masks[a]])
                pairs.setdefault(f'{a}_vs_{b}', []).append(float(score))
                duties.setdefault(f'{a}_vs_{b}', []).append(
                    (float(np.mean([m.mean() for m in masks[a]])),
                     float(np.mean([m.mean() for m in masks[b]]))))
    out = {}
    for key, values in pairs.items():
        out[f'{key}_iou'] = float(np.median(values))
        d = np.asarray(duties[key])
        out[f'{key}_duty_r'] = (float(np.corrcoef(d[:, 0], d[:, 1])[0, 1])
                                if len(d) > 2 else np.nan)
    out['n_cells'] = len(next(iter(pairs.values()))) if pairs else 0
    return out


def corpus_report(predict: PredictFn, corpus_path: str, threshold: float = 0.5,
                  by: Sequence[str] = ('trace_type', 'drug'),
                  wells: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Score a model against a distilled corpus, broken down by provenance.

    The labels in such a corpus are the shipped model's own output on a higher-SNR
    aggregation of the same cell, so this measures **imitation fidelity, not accuracy**. A
    model can only look perfect here by reproducing the teacher, including its mistakes.
    Read it alongside the synthetic scores, which carry exact labels.

    Parameters
    ----------
    predict : PredictFn
    corpus_path : str
        Corpus of real traces, in the layout
        :meth:`~contraction_net.data.ContractionDataset.load_corpus` reads.
    threshold : float, optional
    by : sequence of str, optional
        Metadata columns to break the score down by.
    wells : sequence of str or None, optional
        Restrict to these wells, e.g. the ones held out of training.
    """
    from .data import ContractionDataset

    data = ContractionDataset.load_corpus(corpus_path)
    n = len(data['signals'])
    keep = np.ones(n, dtype=bool)
    if wells is not None and 'well_uid' in data:
        keep = np.isin(data['well_uid'], list(wells))
    frametimes = data.get('frametime', np.full(n, 0.0164))

    scores, duty_pred, duty_true = [], [], []
    for i in range(n):
        if not keep[i]:
            continue
        prob = predict(data['signals'][i], float(frametimes[i]))
        scores.append(iou(prob, data['labels'][i], threshold))
        duty_pred.append(float(_binary(prob, threshold).mean()))
        duty_true.append(float(data['labels'][i].mean()))
    out = {'overall': float(np.mean(scores)) if scores else np.nan,
           'n': len(scores),
           'duty_pred': float(np.mean(duty_pred)) if duty_pred else np.nan,
           'duty_true': float(np.mean(duty_true)) if duty_true else np.nan,
           'CIRCULAR': 'labels are the shipped model on an aggregated signal'}
    idx = np.flatnonzero(keep)
    for column in by:
        if column not in data:
            continue
        values = np.asarray(data[column])[idx]
        out[column] = {str(v): float(np.mean(np.asarray(scores)[values == v]))
                       for v in np.unique(values)}
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _print_grid(grid: Dict[str, np.ndarray]) -> None:
    duties, durations, m = grid['duties'], grid['durations'], grid['iou']
    print('  IoU   duty:' + ''.join(f'{d:7.2f}' for d in duties))
    for i, dur in enumerate(durations):
        cells = ''.join('    -  ' if np.isnan(v) else f'{v:7.2f}' for v in m[i])
        print(f'  dur={dur:4d} fr   {cells}')
    low = m[:, duties <= 0.65]
    high = m[:, duties >= 0.75]
    if low.size and np.isfinite(low).any():
        print(f'  mean IoU at duty <= 0.65: {np.nanmean(low):.3f}')
    if high.size and np.isfinite(high).any():
        print(f'  mean IoU at duty >= 0.75: {np.nanmean(high):.3f}')


def run(model_paths: Sequence[str], repo_root: str = '.', stress_n: int = 300,
        threshold: Optional[float] = None, skip_real: bool = False) -> None:
    """Run every probe for each checkpoint and print a comparison report.

    With ``threshold=None`` each model is scored at the operating point it was tuned for,
    read from its own checkpoint. Comparing two architectures at one shared threshold
    measures the threshold as much as the model.
    """
    stress = make_stress_set(n=stress_n)
    probe_signal, _ = grid_trace(40, 0.45, n_frames=500, seed=0)

    # Bound the benchmark first: a synthetic set can always be made unsolvable, so a model
    # score only means something read against what a simple classical detector achieves on
    # the same traces.
    print('=' * 78)
    print('REFERENCE CEILING  oracle-amplitude detector (bounds the benchmark, not a model)')
    print('=' * 78)
    ref = reference_ceiling(stress)
    print(f'  stress set IoU (event-bearing traces): {ref["overall"]:.3f}   '
          f'(same detector on the noiseless signal: {ref["noiseless"]:.3f} -- the gap below '
          f'that\n   is what the noise costs, the rest is the reference method itself)')
    print(f'  false positives on {ref["n_quiescent"]} quiescent traces: '
          f'{100 * ref["quiescent_fp"]:.1f}% of frames')
    print('  by duty decile: ' + '  '.join(
        f'{lo:.1f}:{v:.2f}' for lo, (_, v) in sorted(ref['by_duty'].items())))
    print('  by frames/contraction: ' + '  '.join(
        f'{lo}{"+" if hi > 10 ** 5 else "-" + str(hi)}:{v:.2f}'
        for (lo, hi), (_, v) in sorted(ref['by_sampling'].items())))
    print()

    for path in model_paths:
        print('=' * 78)
        print(f'MODEL  {path}')
        print('=' * 78)
        predict = model_predictor(path)
        thr = recommended_threshold(path) if threshold is None else threshold
        print(f'  operating point: threshold {thr}'
              + ('  (from the checkpoint)' if threshold is None else '  (overridden)'))

        print('\n[1] duty x duration grid (controlled probes)')
        _print_grid(duty_duration_grid(predict, threshold=thr))

        print('\n[2] sampling sweep (frames resolving one contraction, duty fixed at 0.45)')
        sw = sampling_sweep(predict, threshold=thr)
        print('  frames/event:' + ''.join(f'{int(f):7d}' for f in sw['frames_per_event']))
        print('  IoU         :' + ''.join(
            '    -  ' if np.isnan(v) else f'{v:7.2f}' for v in sw['iou']))

        print(f'\n[3] held-out stress set (n={len(stress)}, disjoint seed, harsh nuisances)')
        sc = score_traces(predict, stress, threshold=thr)
        print(f'  mean IoU (event-bearing traces): {sc["overall"]:.3f}   '
              f'false positives on {sc["n_quiescent"]} quiescent traces: '
              f'{100 * sc["quiescent_fp"]:.1f}% of frames')
        print('  by duty decile:')
        for lo, (n, v) in sorted(sc['by_duty'].items()):
            print(f'    [{lo:.1f},{lo + 0.1:.1f})  n={n:4d}  IoU={v:.3f}  ' + '#' * int(40 * v))
        print('  by regime:')
        for reg, (n, v, ce) in sorted(sc['by_regime'].items()):
            print(f'    {reg:11s} n={n:4d}  IoU={v:.3f}  mean |event-count error|={ce:.1f}')
        print('  by sampling (frames per contraction):')
        for (lo, hi), (n, v) in sorted(sc['by_sampling'].items()):
            span = f'{lo}-{hi}' if hi < 10 ** 6 else f'{lo}+'
            print(f'    {span:>9s} fr  n={n:4d}  IoU={v:.3f}  ' + '#' * int(40 * v))

        print('\n[4] determinism (6 repeats on one trace)')
        det = determinism_report(predict, probe_signal, threshold=thr)
        print(f'  max pointwise spread={det["max_pointwise_spread"]:.4f}  '
              f'unstable frames={100 * det["frac_frames_unstable"]:.1f}%  '
              f'n_contr values={det["n_contr_values"]}')

        print('\n[5] offset invariance (a constant carries no information)')
        for k, v in offset_invariance_report(predict, probe_signal, threshold=thr).items():
            print(f'  {k:12s} -> {100 * v:5.1f}% of frames change label')

        if not skip_real:
            print('\n[6] real recordings')
            rep = real_data_report(predict, repo_root=repo_root, threshold=thr)
            if not rep:
                print('  (no packaged recordings found; pass --repo-root)')
            for name, r in rep.items():
                print(f'  {name}: rows={r["n_rows"]:3d}  duty mean={r["duty_mean"]:.3f} '
                      f'max={r["duty_max"]:.3f}  rows>0.70={100 * r["frac_rows_above_0.70"]:.0f}%  '
                      f'n_contr median={r["n_contr_median"]:.0f}  '
                      f'longest event={r["longest_event_max"]} fr')
        print()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='append', required=True,
                        help='checkpoint to score; repeat to compare models')
    parser.add_argument('--repo-root', default='.', help='root containing test_data/')
    parser.add_argument('--stress-n', type=int, default=300, help='stress-set size')
    parser.add_argument('--threshold', type=float, default=None,
                        help="binarisation threshold; default reads each model's own")
    parser.add_argument('--skip-real', action='store_true', help='skip the real-data probe')
    args = parser.parse_args(argv)
    run(args.model, repo_root=args.repo_root, stress_n=args.stress_n,
        threshold=args.threshold, skip_real=args.skip_real)


if __name__ == '__main__':
    main()
