"""
Synthetic training and benchmark data for ContractionNet.

The generator this module replaces (``utils.simulate_training_data``) built traces from a
clipped cosine, so the contracting fraction of a trace ("duty") was an accidental by-product
of the clip threshold rather than something one could ask for. That is why the shipped
training set tops out at duty 0.78 and contains no contraction longer than 107 frames, and
why the shipped model collapses once a recording spends more than ~70% of its time
contracting.

Here the twitch waveform is parameterised physiologically (time to peak, plateau, relaxation)
and **duty is sampled directly**, so the full range 0.0-0.97 is reachable by construction,
including single tonic contractions and fused beats that never return to baseline.

The second axis is **sampling quality**. The detector is not only for high-speed sarcomere
recordings: it has to work on time-series whose sampling resolves a whole contraction in
three or four points as well as on ones that spread it over hundreds. Frames per
contraction is therefore sampled log-uniformly over :data:`FRAMES_PER_EVENT` and trace
length over 64-4096, rather than being fixed at the 512 the old generator hard-coded.

Sarcomere length is the modelled quantity, so rest is *high* and contraction is *low*.
Amplitudes are in µm and ``frametime`` in seconds, but frametime only scales the velocity
channel -- it must never change the segmentation.

Examples
--------
>>> from contraction_net.simulation import simulate_dataset
>>> traces = simulate_dataset(n=100, seed=0)
>>> tr = traces[0]
>>> tr.signal.shape == tr.label.shape
True
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter

__all__ = [
    'Trace',
    'twitch_waveform',
    'simulate_trace',
    'simulate_dataset',
    'make_stress_set',
    'estimate_noise_params',
    'velocity_channel',
    'REGIMES',
    'FRAMES_PER_EVENT',
    'FRAMETIME_S',
]

#: Contraction regimes the generator can produce.
REGIMES = ('regular', 'tonic', 'fused', 'arrhythmic', 'quiescent')

#: Fraction of peak amplitude below which the clean signal counts as "at rest". The label
#: is derived from the noiseless signal crossing this level, so fused beats that never
#: return to baseline merge into a single contraction interval automatically.
REST_TOL = 0.05

#: Plausible spontaneous beating rates (Hz) for cultured cardiomyocytes. Only used to keep
#: the *seconds*-domain metadata physiological; the quantity that actually governs
#: difficulty is :data:`FRAMES_PER_EVENT`.
BEAT_RATE_HZ = (0.2, 5.0)

#: Frames spanned by a single contraction, sampled log-uniformly. This -- not duration in
#: seconds -- is the axis the model actually sees, and it is the axis the network has to be
#: invariant along: the same detector must work on a coarsely sampled recording that
#: resolves a contraction in 3 frames and on a high-speed one that spreads it over 500.
#: Three frames is the floor at which an excursion is still distinguishable from an
#: outlier; anything less is not resolvable in principle.
FRAMES_PER_EVENT = (3, 600)

#: Frame intervals (s) the generator samples, spanning slow widefield time-lapse through
#: high-speed confocal. Frametime only scales the velocity channel and the seconds-domain
#: metadata -- it must not change the segmentation.
FRAMETIME_S = (0.005, 1.0)

#: How much harsher the ``extreme`` nuisance ranges are than the defaults. Tuned so the
#: benchmark's oracle reference still scores well clear of chance on the stress set: a set
#: that even an oracle cannot segment measures the noise floor, not the model.
HARSH_MULTIPLIER = 1.3

#: Highest duty reachable with beats that relax fully between contractions. Periodic
#: twitches need some diastole, so ``regular`` and ``arrhythmic`` saturate here; higher
#: duty is only physical as a sustained (``tonic``) or fused contraction.
SEPARATED_BEAT_DUTY_MAX = 0.75


@dataclass
class Trace:
    """One simulated time-series and its ground truth.

    Attributes
    ----------
    signal : np.ndarray
        Sarcomere length trace (µm), shape ``(n_frames,)``, including all nuisance effects.
    label : np.ndarray
        Boolean contraction state, shape ``(n_frames,)``, derived from the *clean* signal.
    clean : np.ndarray
        The noiseless signal, kept for debugging and for plotting benchmark failures.
    frametime : float
        Time between frames in s.
    regime : str
        One of :data:`REGIMES`.
    meta : dict
        Sampled parameters, for stratification and for reporting which regime a failure
        came from.
    """

    signal: np.ndarray
    label: np.ndarray
    clean: np.ndarray
    frametime: float
    regime: str
    meta: dict = field(default_factory=dict)

    @property
    def duty(self) -> float:
        """Fraction of frames labelled as contracting."""
        return float(self.label.mean())

    @property
    def n_events(self) -> int:
        """Number of contraction intervals."""
        from scipy.ndimage import label as _label
        return int(_label(self.label)[1])


def twitch_waveform(n_frames: int, rise_frac: float = 0.25, plateau_frac: float = 0.15,
                    relax_k: float = 3.0) -> np.ndarray:
    """
    Single twitch shape, normalised to peak 1 at full contraction and 0 at rest.

    The shape is deliberately asymmetric, like a real cardiomyocyte twitch: a smooth,
    fast shortening phase, an optional plateau, then a slower, roughly exponential
    relaxation.

    Parameters
    ----------
    n_frames : int
        Duration of the twitch in frames.
    rise_frac : float, optional
        Fraction of the twitch spent shortening. Default is 0.25.
    plateau_frac : float, optional
        Fraction of the twitch spent at peak contraction. Default is 0.15.
    relax_k : float, optional
        Exponential rate of the relaxation phase; larger is faster-decaying and more
        strongly curved. Default is 3.0.

    Returns
    -------
    np.ndarray
        Waveform of shape ``(n_frames,)`` with values in ``[0, 1]``.
    """
    n_frames = max(int(n_frames), 3)
    n_rise = max(1, int(round(n_frames * rise_frac)))
    n_plateau = max(0, int(round(n_frames * plateau_frac)))
    n_relax = max(1, n_frames - n_rise - n_plateau)

    # raised cosine: smooth, zero-derivative at both ends, so no kink at onset
    rise = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n_rise, endpoint=False)))
    plateau = np.ones(n_plateau)
    u = np.linspace(0, 1, n_relax)
    relax = (np.exp(-relax_k * u) - np.exp(-relax_k)) / (1 - np.exp(-relax_k))

    return np.concatenate([rise, plateau, relax])[:n_frames]


def _ar1(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance AR(1) process; ``rho`` -> 1 gives slow drift, 0 gives white noise."""
    rho = float(np.clip(rho, 0.0, 0.9999))
    noise = rng.standard_normal(n)
    if rho == 0.0:
        return noise
    out = np.empty(n)
    out[0] = noise[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + noise[i]
    # scale back to unit variance
    return out * np.sqrt(1 - rho ** 2)


def estimate_noise_params(traces: Sequence[np.ndarray], quiet_window: int = 25
                          ) -> Tuple[float, float]:
    """
    Estimate noise scale and correlation from the quietest stretch of each real trace.

    Real sarcomere-length traces do not carry white Gaussian noise: successive frames are
    correlated because the length is derived from smoothly-moving band positions. Training
    on white noise therefore under-prepares the model. This fits an AR(1) description
    (standard deviation and lag-1 autocorrelation) to the flattest window of each input
    trace, which :func:`simulate_trace` then reproduces.

    Parameters
    ----------
    traces : sequence of np.ndarray
        Real 1D traces to estimate from.
    quiet_window : int, optional
        Length in frames of the window searched for the flattest stretch. Default is 25.

    Returns
    -------
    tuple of float
        ``(sigma_relative, rho)`` -- noise standard deviation as a fraction of the trace's
        own robust amplitude, and the lag-1 autocorrelation of the residual.
    """
    sigmas, rhos = [], []
    for tr in traces:
        tr = np.asarray(tr, dtype=float)
        tr = tr[np.isfinite(tr)]
        if tr.size < 3 * quiet_window:
            continue
        # flattest window = smallest rolling variance of the first difference
        d = np.diff(tr)
        csum = np.concatenate([[0.0], np.cumsum(d ** 2)])
        w = quiet_window
        roll = csum[w:] - csum[:-w]
        i = int(np.argmin(roll))
        seg = tr[i:i + w + 1]
        # detrend so slow drift is not counted as noise
        seg = seg - np.polyval(np.polyfit(np.arange(seg.size), seg, 1), np.arange(seg.size))
        amp = np.percentile(tr, 95) - np.percentile(tr, 5)
        if amp <= 0 or seg.std() <= 0:
            continue
        sigmas.append(seg.std() / amp)
        r = np.corrcoef(seg[:-1], seg[1:])[0, 1]
        if np.isfinite(r):
            rhos.append(np.clip(r, 0.0, 0.95))
    if not sigmas:
        return 0.05, 0.5
    return float(np.median(sigmas)), float(np.median(rhos))


def velocity_channel(signal: np.ndarray, frametime: float, window_length: int = 13,
                     polyorder: int = 5) -> np.ndarray:
    """
    Contraction velocity, the second network input channel.

    Uses the same Savitzky-Golay derivative that SarcAsM already applies downstream in
    :func:`sarcasm.analysis.contraction_analysis.analyze_contraction_parameters`, so the
    channel the model trains on is the one the pipeline produces.

    Unlike length, velocity has a genuinely zero-centred baseline: it is near zero whenever
    the cell is at rest *and* whenever it is held in a sustained contraction, with clear
    excursions only at onset and offset. That makes it far less sensitive to how much of a
    recording is spent contracting.

    Parameters
    ----------
    signal : np.ndarray
        1D sarcomere length trace.
    frametime : float
        Time between frames in s.
    window_length : int, optional
        Savitzky-Golay window in frames. Default is 13.
    polyorder : int, optional
        Savitzky-Golay polynomial order. Default is 5.

    Returns
    -------
    np.ndarray
        Velocity trace (µm/s) of the same shape as ``signal``.
    """
    signal = np.asarray(signal, dtype=float)
    # window must be odd, <= len(signal), and > polyorder
    w = min(int(window_length), signal.size if signal.size % 2 else signal.size - 1)
    if w % 2 == 0:
        w -= 1
    if w <= polyorder or w < 3:
        return np.gradient(signal, frametime)
    return savgol_filter(signal, w, polyorder, deriv=1, delta=frametime)


def _labelled_fraction(shape: np.ndarray) -> float:
    """Fraction of a twitch waveform that clears :data:`REST_TOL` and is therefore labelled.

    The onset and the tail of the relaxation sit below the rest tolerance, so a twitch of
    ``d`` frames yields fewer than ``d`` labelled frames. Duty is a requested parameter
    here, so the event duration has to be solved for rather than set to ``duty * period``
    directly -- otherwise every regime silently undershoots its target.
    """
    return max(float((shape > REST_TOL).mean()), 1e-3)


def _event_frames(duty: float, period: int, shape_fracs: Tuple[float, float, float]) -> int:
    """Twitch duration in frames that yields ``duty`` after rest-tolerance trimming."""
    probe = twitch_waveform(256, *shape_fracs)
    d = duty * period / _labelled_fraction(probe)
    return int(np.clip(round(d), 3, period))


def _sample_period(duty: float, n_frames: int, rng: np.random.Generator) -> int:
    """Beat period in frames, drawn so the contraction spans a resolvable number of frames.

    Sampled from :data:`FRAMES_PER_EVENT` rather than from a beating rate in Hz: the model
    has to generalise across sampling quality, so the frames-per-contraction axis is the
    one that must be covered uniformly (in log space). Deriving frames from a physiological
    rate instead would leave the coarse end -- a contraction resolved by 3 or 4 samples --
    almost unpopulated at high frame rates.
    """
    lo, hi = FRAMES_PER_EVENT
    frames_per_event = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    period = int(round(frames_per_event / max(duty, 1e-6)))
    return int(np.clip(period, max(4, int(np.ceil(lo / max(duty, 1e-6)))), max(8, n_frames)))


def _label_from_clean(clean: np.ndarray, baseline: float, amplitude: float) -> np.ndarray:
    """Contraction state derived from the noiseless signal.

    Deriving the label from the clean signal rather than from the event schedule means
    fused beats, which never return to baseline, merge into one interval automatically --
    the label always matches what is actually visible in the trace.
    """
    if amplitude <= 0:
        return np.zeros(clean.size, dtype=bool)
    return (baseline - clean) > REST_TOL * amplitude


def _apply_nuisances(clean, rng, *, noise_rel, noise_rho, drift_rel,
                     bleach_rel, n_steps, step_rel, n_outliers, outlier_rel,
                     quantise_rel, gap_frac, amplitude, artefact_amp=None):
    """Add the acquisition and tracking artefacts that real traces carry.

    ``artefact_amp`` scales everything except the measurement noise. It is separate from
    ``amplitude`` so that quiescent traces, which have no contraction amplitude to
    reference, get baseline artefacts sized against their noise rather than against a
    contraction that is not there.
    """
    n = clean.size
    sig = clean.copy()
    if artefact_amp is None:
        artefact_amp = amplitude

    # correlated measurement noise
    if noise_rel > 0:
        sig = sig + noise_rel * amplitude * _ar1(n, noise_rho, rng)

    # Slow baseline wander. This has to be genuinely *slow*: an unconstrained random walk
    # of comparable amplitude produces contraction-shaped excursions, which would make the
    # ground truth ambiguous -- a drifting quiescent trace and a slow tonic contraction are
    # the same signal. Two low-frequency sinusoids with periods of at least half the trace
    # give wander that no detector can mistake for a twitch.
    if drift_rel > 0:
        t = np.linspace(0, 1, n)
        sig = sig + drift_rel * artefact_amp * (
            np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * t + rng.uniform(0, 2 * np.pi))
            + 0.5 * np.sin(2 * np.pi * rng.uniform(0.3, 2.0) * t + rng.uniform(0, 2 * np.pi))
        ) / 1.5

    # photobleaching-like monotonic trend
    if bleach_rel != 0:
        sig = sig + bleach_rel * artefact_amp * np.linspace(0, 1, n)

    # baseline steps from track re-acquisition or sarcomere popping
    for _ in range(int(n_steps)):
        i = rng.integers(1, n)
        sig[i:] += step_rel * artefact_amp * rng.choice([-1.0, 1.0])

    # isolated outlier frames
    for _ in range(int(n_outliers)):
        i = rng.integers(0, n)
        sig[i] += outlier_rel * artefact_amp * rng.standard_normal()

    # sarcomere length comes from discrete band positions, so it can be step-like
    if quantise_rel > 0:
        step = quantise_rel * artefact_amp
        sig = np.round(sig / step) * step

    # NaN gaps, then the same linear fill the inference path applies via
    # Motion._fill_trace_nans -- real inputs contain these piecewise-linear stretches
    if gap_frac > 0:
        n_gap = int(round(gap_frac * n))
        if n_gap >= 1:
            mask = np.zeros(n, dtype=bool)
            remaining = n_gap
            while remaining > 0:
                length = int(min(remaining, rng.integers(2, max(3, n // 20))))
                start = int(rng.integers(0, max(1, n - length)))
                mask[start:start + length] = True
                remaining -= length
            if mask.any() and not mask.all():
                idx = np.arange(n)
                sig[mask] = np.interp(idx[mask], idx[~mask], sig[~mask])

    return sig


def simulate_trace(regime: str = 'regular', *, duty: Optional[float] = None,
                   n_frames: Optional[int] = None, frametime: Optional[float] = None,
                   rng: Optional[np.random.Generator] = None,
                   noise_rel: float = 0.05, noise_rho: float = 0.5,
                   amplitude: Optional[float] = None, extreme: bool = False) -> Trace:
    """
    Simulate one sarcomere-length trace with exact ground truth.

    Parameters
    ----------
    regime : str, optional
        One of :data:`REGIMES`. Default is ``'regular'``.

        - ``'regular'``: periodic beating at the requested duty.
        - ``'tonic'``: a single sustained contraction filling most of the recording -- the
          case the shipped model fails on.
        - ``'fused'``: beats whose relaxation is truncated, so the trace never returns to
          baseline between them (summation). Absent from the shipped training set.
        - ``'arrhythmic'``: irregular intervals, ectopic beats, pauses and alternans.
        - ``'quiescent'``: no contraction at all.
    duty : float or None, optional
        Target fraction of frames spent contracting, in ``[0, 0.97]``. Sampled uniformly
        if None. Values above 0.97 are refused: a trace that is entirely contracted, with
        no visible transition, carries no absolute reference and is genuinely undecidable
        for an amplitude-normalised model.
    n_frames : int or None, optional
        Trace length. Sampled log-uniformly from 64-4096 if None.
    frametime : float or None, optional
        Time between frames in s. Sampled from 5-200 Hz if None, so the model learns to be
        frame-rate invariant.
    rng : np.random.Generator or None, optional
        Random generator. A fresh default generator is used if None.
    noise_rel : float, optional
        Noise standard deviation as a fraction of contraction amplitude. Default is 0.05.
    noise_rho : float, optional
        Lag-1 autocorrelation of the noise. Default is 0.5. See
        :func:`estimate_noise_params`.
    amplitude : float or None, optional
        Contraction amplitude in µm. Sampled if None.
    extreme : bool, optional
        Push the nuisance parameters to the edge of their ranges. Used to build the
        held-out stress set. Default is False.

    Returns
    -------
    Trace
        The simulated trace, its ground truth and the sampled parameters.

    Raises
    ------
    ValueError
        If ``regime`` is unknown or ``duty`` is outside ``[0, 0.97]``.
    """
    if regime not in REGIMES:
        raise ValueError(f'Unknown regime {regime!r}; expected one of {REGIMES}.')
    if rng is None:
        rng = np.random.default_rng()
    if duty is not None and not 0.0 <= duty <= 0.97:
        raise ValueError(f'duty must lie in [0, 0.97], got {duty}. A fully contracted '
                         'trace has no visible transition and no absolute reference, so '
                         'it is not decidable.')

    # Log-uniform, and starting well below the 512 the old generator hard-coded: short
    # coarsely-sampled recordings are exactly the regime the detector has to generalise to,
    # and they are absent from both the shipped training set and real high-speed data.
    n_frames = (int(n_frames) if n_frames is not None
                else int(round(np.exp(rng.uniform(np.log(64), np.log(4096))))))
    frametime = (float(frametime) if frametime is not None
                 else float(np.exp(rng.uniform(*np.log(FRAMETIME_S)))))
    amplitude = float(amplitude) if amplitude is not None else float(rng.uniform(0.05, 0.6))
    baseline = float(rng.uniform(1.6, 2.3))

    if regime == 'quiescent':
        duty = 0.0
    elif duty is None:
        duty = float(rng.uniform(0.0, 0.97))

    rise = float(rng.uniform(0.15, 0.40))
    plateau = float(rng.uniform(0.0, 0.35))
    relax_k = float(rng.uniform(1.5, 5.0))

    clean = np.full(n_frames, baseline)
    meta = {'baseline': baseline, 'amplitude': amplitude, 'rise_frac': rise,
            'plateau_frac': plateau, 'relax_k': relax_k, 'duty_target': duty}

    if regime == 'quiescent':
        pass

    elif regime == 'tonic':
        # one sustained contraction; duty controls how much of the trace it fills.
        # A tonic contraction is mostly plateau.
        fracs = (min(rise, 0.2), max(0.4, 1 - 2 * rise), relax_k)
        d = min(_event_frames(duty, n_frames, fracs), n_frames - 2)  # leave a transition
        if d < FRAMES_PER_EVENT[0]:
            meta['n_events_scheduled'] = 0     # not resolvable at this sampling
        else:
            start = int(rng.integers(0, max(1, n_frames - d)))
            clean[start:start + d] -= amplitude * twitch_waveform(d, *fracs)
            meta['n_events_scheduled'] = 1

    elif regime in ('regular', 'fused'):
        fracs = (rise, plateau, relax_k)
        # The beat period follows from how many frames should resolve one contraction, so
        # sampling quality is covered uniformly. Deriving it from a beating rate in Hz
        # instead leaves the coarse end almost unpopulated at high frame rates, and the
        # detector has to work on series that resolve a contraction in a handful of samples.
        period = _sample_period(duty, n_frames, rng)
        # a beat needs a resolvable number of frames: at very low duty the answer is fewer
        # beats, not shorter ones, or duty is silently inflated
        if n_frames // period < 1 or duty * period < FRAMES_PER_EVENT[0]:
            meta['n_events_scheduled'] = 0
        else:
            meta['beat_rate_hz'] = 1 / (period * frametime)
            fusion = float(rng.uniform(0.3, 0.9)) if regime == 'fused' else 0.0
            if regime == 'fused':
                # the fused run itself occupies `duty` of the trace; inside it the beats
                # sit shoulder to shoulder on a sustained pedestal
                run = int(np.clip(round(duty * n_frames), 8, n_frames - 2))
                first = int(rng.integers(0, max(1, n_frames - run)))
                # beats inside the run keep the sampled period, so a fused burst is
                # resolved as coarsely or as finely as any other regime
                d = int(np.clip(period, FRAMES_PER_EVENT[0], max(FRAMES_PER_EVENT[0], run)))
                starts = list(range(first, first + run - 3, d))
                last = min(first + run, n_frames)

                # Summation is a pedestal with twitches on top, not a raised waveform:
                # the first onset still rises from rest, and only the relaxation between
                # beats is prevented from returning to baseline.
                pedestal = np.zeros(n_frames)
                pedestal[first:last] = 1.0
                edge = max(1, int(round(0.15 * (last - first))))
                ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, edge)))
                pedestal[first:first + edge] = ramp[:max(0, min(edge, last - first))]
                tail = max(first, last - edge)
                pedestal[tail:last] = ramp[::-1][:last - tail]
                clean -= amplitude * fusion * pedestal
            else:
                d = _event_frames(duty, period, fracs)
                starts = list(range(int(rng.integers(0, max(1, period))), n_frames - 3, period))

            shape = twitch_waveform(d, rise_frac=rise, plateau_frac=plateau, relax_k=relax_k)
            for start in starts:
                seg = shape[:min(d, n_frames - start)]
                clean[start:start + seg.size] -= amplitude * (1 - fusion) * seg
            meta.update(n_events_scheduled=len(starts), period=period, fusion=fusion)

    else:  # arrhythmic
        fracs = (rise, plateau, relax_k)
        mean_period = _sample_period(duty, n_frames, rng)
        min_d = FRAMES_PER_EVENT[0]
        if n_frames // mean_period < 1 or duty * mean_period < min_d:
            meta['n_events_scheduled'] = 0
        else:
            meta['beat_rate_hz'] = 1 / (mean_period * frametime)
            d_mean = _event_frames(duty, mean_period, fracs)
            pos, scheduled = int(rng.integers(0, mean_period)), 0
            while pos < n_frames - 3:
                # jitter is normalised to mean 1 and the pause budget is taken out of it,
                # so irregularity scatters the intervals without dragging duty down
                jitter = float(rng.uniform(0.7, 1.3))
                d = max(min_d, int(round(d_mean * rng.uniform(0.7, 1.3))))
                if rng.random() < 0.12:      # ectopic beat: early and small
                    amp_i, d = amplitude * rng.uniform(0.3, 0.6), max(min_d, d // 2)
                    jitter = 0.4
                elif rng.random() < 0.12:    # alternans: alternating amplitude
                    amp_i = amplitude * (0.6 if scheduled % 2 else 1.0)
                else:
                    amp_i = amplitude
                shape = twitch_waveform(d, rise_frac=rise, plateau_frac=plateau,
                                        relax_k=relax_k)
                seg = shape[:min(d, n_frames - pos)]
                clean[pos:pos + seg.size] -= amp_i * seg
                scheduled += 1
                step = int(round(d / max(duty, 1e-3) * jitter))
                if rng.random() < 0.08:      # pause
                    step *= int(rng.integers(2, 4))
                pos += max(d, step)
            meta['n_events_scheduled'] = scheduled

    label = _label_from_clean(clean, baseline, amplitude)

    # On a trace with no contraction there is no contraction amplitude to reference, so
    # scaling baseline artefacts by the sampled one would inject drift and steps many times
    # the noise -- excursions that look exactly like contractions in a trace labelled as
    # having none. Size them against the noise instead, so a quiescent trace really is flat
    # plus noise and its ground truth is unambiguous.
    noise_eff = min(noise_rel * (HARSH_MULTIPLIER if extreme else 1.0), 0.35)
    artefact_amp = amplitude if label.any() else amplitude * noise_eff * 0.5

    # The stress set is meant to be hard, not impossible: noise is allowed to reach ~35% of
    # contraction amplitude, beyond which the ground truth stops being recoverable from the
    # trace at all and the benchmark would measure the noise floor rather than the model.
    harsh = HARSH_MULTIPLIER if extreme else 1.0
    signal = _apply_nuisances(
        clean, rng,
        noise_rel=min(noise_rel * rng.uniform(0.3, 2.0) * harsh, 0.35),
        noise_rho=float(np.clip(noise_rho * rng.uniform(0.5, 1.5), 0, 0.95)),
        drift_rel=float(rng.uniform(0, 0.3 * harsh)),
        bleach_rel=float(rng.uniform(-0.3, 0.3) * harsh),
        n_steps=int(rng.integers(0, 6 if extreme else 3)),
        step_rel=float(rng.uniform(0.05, 0.4 * harsh)),
        n_outliers=int(rng.integers(0, 20 if extreme else 8)),
        outlier_rel=float(rng.uniform(0.2, 1.0 * harsh)),
        quantise_rel=float(rng.uniform(0, 0.15)) if rng.random() < 0.3 else 0.0,
        gap_frac=float(rng.uniform(0, 0.25 if extreme else 0.15)) if rng.random() < 0.4 else 0.0,
        amplitude=amplitude, artefact_amp=artefact_amp,
    )

    meta['duty_actual'] = float(label.mean())
    return Trace(signal=signal, label=label, clean=clean,
                 frametime=frametime, regime=regime, meta=meta)


#: Sampling weights per regime. ``regular`` dominates because that is what most recordings
#: look like; ``tonic`` and ``fused`` are over-represented relative to reality because they
#: are the regimes the shipped model fails on and real data cannot supply them.
REGIME_WEIGHTS = {'regular': 0.34, 'tonic': 0.22, 'fused': 0.20,
                  'arrhythmic': 0.16, 'quiescent': 0.08}


def simulate_dataset(n: int = 1000, seed: int = 0,
                     regimes: Optional[Sequence[str]] = None,
                     noise_rel: float = 0.05, noise_rho: float = 0.5,
                     stratify_duty: bool = True, extreme: bool = False) -> List[Trace]:
    """
    Simulate a dataset stratified over duty cycle and regime.

    Stratification is the point: the real training traces all come from normally-beating
    cells and cluster around duty 0.4-0.5, so the synthetic half has to carry the whole
    range -- especially duty > 0.7, where the shipped model fails.

    Parameters
    ----------
    n : int, optional
        Number of traces. Default is 1000.
    seed : int, optional
        Random seed. Default is 0.
    regimes : sequence of str or None, optional
        Restrict to these regimes, sampled uniformly. Default (None) samples all of
        :data:`REGIMES` with :data:`REGIME_WEIGHTS`.
    noise_rel, noise_rho : float, optional
        Noise scale and correlation; see :func:`estimate_noise_params`.
    stratify_duty : bool, optional
        Spread duty uniformly over ``[0, 0.97]`` rather than sampling it independently.
        Default is True.
    extreme : bool, optional
        Push nuisance parameters to their limits. Default is False.

    Returns
    -------
    list of Trace
    """
    rng = np.random.default_rng(seed)
    allowed = list(REGIME_WEIGHTS) if regimes is None else list(regimes)

    def pick(duty_target):
        # Periodic beats that relax fully cannot exceed duty ~0.8 -- there has to be some
        # diastole between them. Above that the only physical options are a sustained
        # contraction or fused beats, so route there instead of asking `regular` for a
        # duty it cannot produce and silently landing back at 0.75.
        pool = allowed
        if duty_target is not None and duty_target > SEPARATED_BEAT_DUTY_MAX:
            high = [r for r in allowed if r in ('tonic', 'fused')]
            pool = high or allowed
        w = np.array([REGIME_WEIGHTS.get(r, 1.0) for r in pool], dtype=float)
        return str(rng.choice(pool, p=w / w.sum()))

    traces = []
    for i in range(int(n)):
        if stratify_duty:
            # sweep the range in 20 slots, jittered within each, so every duty decile is
            # populated instead of clustering where the sampler happens to land
            duty = float(np.clip((i % 20 + rng.random()) / 20 * 0.97, 0.0, 0.97))
        else:
            duty = None
        regime = pick(duty)
        if regime == 'quiescent':
            duty = 0.0
        traces.append(simulate_trace(regime, duty=duty, rng=rng, noise_rel=noise_rel,
                                     noise_rho=noise_rho, extreme=extreme))
    return traces


def make_stress_set(n: int = 400, seed: int = 99999) -> List[Trace]:
    """
    Held-out benchmark set, generated with a different seed and harsher parameters.

    Scoring on traces drawn from the training generator would reward memorising the
    generator. This set uses a disjoint seed and pushes noise, drift, steps and outliers to
    the edge of their ranges, so a good score means the model generalises within the
    regime rather than having learned the simulator.

    Parameters
    ----------
    n : int, optional
        Number of traces. Default is 400.
    seed : int, optional
        Random seed, deliberately far from the training seeds. Default is 99999.

    Returns
    -------
    list of Trace
    """
    return simulate_dataset(n=n, seed=seed, noise_rel=0.07, noise_rho=0.7,
                            stratify_duty=True, extreme=True)
