import math
import os
import threading
import warnings

import numpy as np
import torch
from scipy.signal import savgol_filter

from .contraction_net import ContractionNet, SymmetrizedContractionNet
from .padding import INFERENCE_PAD, pad_for_inference
from .polarity import PolarityDecision, PolarityRule, decide_polarity

#: Device inference runs on. The CPU by default: ``predict_contractions(-x)`` equals
#: ``predict_contractions(x)`` bit for bit only if two calls on the same input agree, which
#: is verified on the CPU alone (``TestMirrorInvariance`` in tests/test_contraction_net.py).
#: To use another device, set ``contraction_net.prediction.device`` to it, e.g.
#: ``torch.device('cuda:0')``, and first run that test with ``CONTRACTION_NET_DEVICE`` set
#: to the same device. Training picks its own device with
#: :func:`contraction_net.utils.get_device`.
device = torch.device('cpu')

# Loaded models, keyed by (path, mtime, device); prediction is called once per
# myofibril/domain, so re-reading the checkpoint each time dominated the runtime.
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()

#: Frametime assumed by the polarity rule when the caller gives none: CardioTrace's 30 fps.
_DEFAULT_FRAMETIME = 1 / 30

#: Architectures a checkpoint may name in its ``arch`` field.
_ARCHS = {'ContractionNet': ContractionNet,
          'SymmetrizedContractionNet': SymmetrizedContractionNet}

_PRE_1_0_MESSAGE = (
    '{model} predates the 1.0 ContractionNet: its checkpoint records no architecture, so '
    'the weights cannot be matched to a network. The pre-1.0 model coupled its '
    'normalisation across the whole time axis and lost contractions in recordings that '
    'spend most of their time contracting. Retrain with contraction_net.Trainer -- see the '
    'ContractionNet training tutorial -- or use the bundled model_ContractionNet.pt.')


def _resolve_network(state, model, network):
    """Pick the network class a checkpoint was trained with.

    ``arch`` was introduced in 1.0, so its absence identifies a pre-1.0 checkpoint.
    """
    arch = state.get('arch')
    if arch is None:
        raise ValueError(_PRE_1_0_MESSAGE.format(model=model))
    return _ARCHS.get(arch, network)


def _load_model(model, network):
    """Load a checkpoint into an eval-mode model, caching by path, mtime and device."""
    if isinstance(model, torch.nn.Module):
        return model.eval()

    try:
        key = (os.fspath(model), os.path.getmtime(model), str(device))
    except (TypeError, OSError):
        key = None

    if key is not None:
        with _CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

    state_dict = torch.load(model, map_location=device, weights_only=True)
    cls = _resolve_network(state_dict, model, network)
    net = cls(state_dict['n_filter'], in_channels=state_dict['in_channels'],
              out_channels=state_dict['out_channels'],
              **(state_dict.get('arch_kwargs') or {})).to(device)
    net.load_state_dict(state_dict['state_dict'])
    # Dropout must not fire at inference, or every call becomes a random ensemble member.
    net.eval()
    # Checkpoints are self-describing: inference reads its own conventions off the file.
    net.input_convention = state_dict.get('input_convention', 'q90')
    net.recommended_threshold = state_dict.get('recommended_threshold', 0.5)
    net.expected_in_channels = int(state_dict['in_channels'])
    net.expected_out_channels = int(state_dict['out_channels'])
    net.polarity_rule = PolarityRule.from_checkpoint(state_dict.get('polarity_rule'))
    net.decoding = state_dict.get('decoding')

    if key is not None:
        with _CACHE_LOCK:
            _MODEL_CACHE[key] = net
    return net


#: Input conditioning conventions understood by :func:`prepare_robust_input`.
INPUT_CONVENTIONS = ('q90', 'symmetric', 'raw')


def prepare_robust_input(data, diff_window=5, convention='q90'):
    """Condition a 1D trace into the two-channel input used by :class:`ContractionNet`.

    Shared by training and inference so the two cannot drift apart.

    Parameters
    ----------
    data : ndarray
        1D time-series, finite.
    diff_window : int, optional
        Smoothing window (frames) for the difference channel. Default is 5.
    convention : str, optional
        ``'q90'`` centres on the 90th percentile: rest is assumed to be the high side of
        the trace, so a signal resting low must be inverted first, which
        :func:`predict_contractions` does by the polarity rule. This is the default.
        ``'symmetric'`` centres on ``(P10 + P90) / 2``, which makes the conditioning exactly
        odd -- ``x -> -x`` maps level to ``-level`` and diff to ``-diff``, since
        ``P10(-x) = -P90(x)`` -- so a polarity-invariant model can be trained with sign-flip
        augmentation. Rest is then not distinguished by sign and must be inferred from the
        waveform.
        ``'raw'`` passes the trace through unchanged as a single channel, for a model
        trained on unconditioned traces, as the pre-1.0 recipe trained it.

    Returns
    -------
    ndarray
        Array of shape ``(2, len(data))``: the referenced level, and its smoothed per-frame
        difference on the same scale. ``'raw'`` returns ``(1, len(data))``.

    Notes
    -----
    Both the reference and the scale are affine-equivariant, so the same trace in µm and in
    nm produces the same output up to rounding, but not bit for bit: the subtraction and the
    division round differently at different scales and offsets. The differences are
    absolute, of order 1e-6 on the output's unit scale and largest where a large offset
    meets a small spread, so a value near zero can even change sign.
    """
    if convention not in INPUT_CONVENTIONS:
        raise ValueError(f'convention must be one of {INPUT_CONVENTIONS}, got {convention!r}')
    x = np.asarray(data, dtype=np.float64).ravel()
    if convention == 'raw':
        return x[None].astype(np.float32)
    q90 = np.percentile(x, 90)
    scale = float(q90 - np.percentile(x, 10))
    rest = q90 if convention == 'q90' else q90 - 0.5 * scale
    if scale <= 0:
        # flat or heavily quantised: fall back to the spread of what variation there is
        scale = float(np.abs(x - rest).max())
    if scale <= 0:
        scale = 1.0

    level = (x - rest) / scale
    if x.size >= 3:
        k = int(min(max(3, diff_window), x.size if x.size % 2 else x.size - 1))
        if k % 2 == 0:
            k -= 1
        # local linear slope per frame; delta=1 keeps the channel free of any frametime
        # dependence, so acquisition rate cannot leak into the model
        diff = savgol_filter(level, k, min(2, k - 1), deriv=1, delta=1.0)
    else:
        diff = np.gradient(level) if x.size > 1 else np.zeros_like(level)
    return np.stack([level, diff]).astype(np.float32)


def predict_contractions(data, model, network=ContractionNet, *, frametime=None, polarity=None,
                         margin=None, return_decision=False):
    """Predict contraction intervals in a time-series with a neural network.

    Parameters
    ----------
    data : ndarray
        1D array with the time-series to analyze.
    model : str or torch.nn.Module
        Trained model weights (.pt file), or an already-constructed module.
    network : nn.Module, optional
        Network class used to instantiate the weights. Default is
        :class:`~contraction_net.contraction_net.ContractionNet`.
    frametime : float, optional
        Time between frames in s. The polarity rule of ``'q90'`` checkpoints measures its
        windows in seconds, so pass it: None assumes 1/30 s and warns. Unused by other
        checkpoints.
    polarity : {None, 1, -1}, optional
        None (default) lets the polarity rule decide which side of the trace is rest. +1 or
        -1 declares it (+1: rest high), for a quantity whose sign physics fixes; the trace
        is then oriented by the declaration and the decision records whether the data
        agree. Only for ``'q90'`` checkpoints.
    margin : float, optional
        Abstention margin on the polarity score. Default is the checkpoint's rule (0.10).
        0 decides every trace whose score is not exactly 0.
    return_decision : bool, optional
        Also return the :class:`~contraction_net.polarity.PolarityDecision`. Default False.

    Returns
    -------
    ndarray
        Array of shape ``(out_channels, len(data))`` with per-frame probabilities. Channel
        0 is the contraction state. float32 zeros when the polarity is undetermined.
    PolarityDecision
        Only with ``return_decision=True``.

    Raises
    ------
    ValueError
        If the trace is empty or non-finite, if ``model`` predates the 1.0 architecture, or
        if a polarity is declared for a checkpoint that does not use the polarity rule.

    Notes
    -----
    The input is conditioned by :func:`prepare_robust_input` using the convention recorded
    in the checkpoint, so it is offset- and scale-invariant.

    A ``'q90'`` checkpoint reads rest as the high side of its input, but the trace no
    longer has to rest high: :func:`~contraction_net.polarity.decide_polarity` decides once
    per trace, on the raw float64 trace, and the trace is multiplied by that sign before
    conditioning. ``x`` and ``-x`` therefore reach the network as the same array, and the
    result is identical bit for bit. A trace that is too short, flat or ambiguous to decide
    gets zeros, never the reading of the wrong side; ``return_decision`` says why.

    ``'symmetric'`` and ``'raw'`` checkpoints keep their own path unchanged: the rule is
    not applied, ``frametime`` and ``margin`` are unused, and the decision record says so.

    The network runs on :data:`device` (the CPU by default) with a batch of one, in eval
    mode.
    """
    net = _load_model(model, network)
    data = _validated_trace(data)
    frametime = _resolve_frametime(net, frametime, 'predict_contractions')
    prepared, decision = _condition(net, data, frametime, polarity, margin)
    if prepared is None:
        probs = np.zeros((_out_channels(net), data.shape[0]), dtype=np.float32)
    else:
        probs = _network_output(net, [prepared], data.shape[0])[0]
    return (probs, decision) if return_decision else probs


def predict_contractions_batch(traces, model, network=ContractionNet, *, frametime=None,
                               polarity=None, margin=None):
    """:func:`predict_contractions` for several traces of one length, in batched network calls.

    Parameters
    ----------
    traces : array_like
        Time-series of shape ``(n_traces, n_frames)``.
    model, network, frametime, polarity, margin
        As for :func:`predict_contractions`; they apply to every trace.

    Returns
    -------
    ndarray
        Array of shape ``(n_traces, out_channels, n_frames)``, row *i* equal to
        ``predict_contractions(traces[i], ...)`` up to float32 rounding (~1e-6).
    list of PolarityDecision
        One per trace, identical to the single call's.

    Notes
    -----
    Each trace is decided and conditioned on its own, and the network treats the traces of
    a batch independently (its normalization and attention act per trace), so batching
    changes only the speed and the last float32 bits: the CPU kernels sum in a different
    order for a batch. Batches hold at most 8 traces, fewer if their attention maps would
    exceed ~256 MB. On an M1 Pro, 8 traces per call were 1.4-2x faster than one at a time
    for 150-3000 frames, and 16 per call 5-15x slower.
    """
    net = _load_model(model, network)
    traces = [_validated_trace(t) for t in np.atleast_2d(np.asarray(traces, dtype=np.float64))]
    n_frames = traces[0].shape[0]
    frametime = _resolve_frametime(net, frametime, 'predict_contractions_batch')
    probs = np.zeros((len(traces), _out_channels(net), n_frames), dtype=np.float32)
    conditioned = [_condition(net, t, frametime, polarity, margin) for t in traces]
    decisions = [decision for _, decision in conditioned]
    run = [i for i, (prepared, _) in enumerate(conditioned) if prepared is not None]

    padded = n_frames + 2 * INFERENCE_PAD
    batch_size = max(1, min(_MAX_BATCH, _ATTENTION_BUDGET // (16 * padded * padded)))
    for start in range(0, len(run), batch_size):
        chunk = run[start:start + batch_size]
        probs[chunk] = _network_output(net, [conditioned[i][0] for i in chunk], n_frames)
    return probs, decisions


#: Traces per batched network call; larger batches were several times slower on the CPU.
_MAX_BATCH = 8
#: Bytes the attention maps of one batched network call may take (float32, 4 heads).
_ATTENTION_BUDGET = 256 * 2 ** 20


def _validated_trace(data):
    data = np.array(data, dtype=np.float64).ravel()
    if data.shape[0] == 0:
        raise ValueError('Cannot predict contractions for an empty time-series.')
    if not np.isfinite(data).all():
        raise ValueError('Time-series contains NaN or infinite values; fill gaps first.')
    return data


def _resolve_frametime(net, frametime, caller):
    """The frametime the polarity rule decides with; warns (at the caller's caller) if none."""
    if frametime is None and getattr(net, 'input_convention', 'q90') == 'q90':
        warnings.warn(f'{caller} was given no frametime; the polarity rule assumes 1/30 s. '
                      f'Pass the frametime of the recording.', stacklevel=3)
        return _DEFAULT_FRAMETIME
    return frametime


def _condition(net, data, frametime, polarity, margin):
    """Decide a trace's polarity and condition it for ``net``: ``(input or None, decision)``.

    The input is None when the polarity rule cannot decide; the prediction is then zeros.
    """
    convention = getattr(net, 'input_convention', 'q90')
    if convention != 'q90':
        if polarity is not None:
            raise ValueError(f'polarity can only be declared for checkpoints that use the '
                             f"polarity rule ('q90'); this one uses {convention!r}.")
        decision = PolarityDecision(1, math.nan, math.nan,
                                    f'rule not used: {convention!r} checkpoint')
        return prepare_robust_input(data, convention=convention), decision

    decision = decide_polarity(data, frametime, polarity=polarity, margin=margin,
                               rule=polarity_rule(net))
    if decision.sign == 0:
        return None, decision
    return prepare_robust_input(decision.sign * data, convention='q90'), decision


def _network_output(net, prepared, len_data):
    """Run ``net`` on conditioned traces of one length as one batch, and crop the padding off again."""
    # a model trained on the level alone must not be handed the difference channel
    want = int(getattr(net, 'expected_in_channels', prepared[0].shape[0]))
    # pad so the edge frames are inferred from context, not from a truncated window
    padded = [pad_for_inference(p[:want]) for p in prepared]
    pad = padded[0][1]

    tensor = torch.from_numpy(np.stack([p for p, _ in padded])).to(device)
    with torch.no_grad():
        res = net(tensor)[0]
    res = res.detach().cpu().numpy()

    return res[:, :, pad:pad + len_data]


def _out_channels(net):
    """Number of channels ``net`` outputs, found without running it.

    A checkpoint records the count. A bare module is read off its output layer: the head of
    a :class:`ContractionNet`, or of the trunk of a :class:`SymmetrizedContractionNet`.
    """
    recorded = getattr(net, 'expected_out_channels', None)
    if recorded is not None:
        return int(recorded)
    trunk = getattr(net, 'trunk', None)
    for layer in (getattr(net, 'head', None), getattr(trunk, 'head', None)):
        if isinstance(layer, torch.nn.Conv1d):
            return int(layer.out_channels)
    raise ValueError(f'Cannot tell how many channels {type(net).__name__} outputs; load it '
                     f'from a checkpoint, or set its expected_out_channels.')


def recommended_threshold(model, default=0.5):
    """Decision threshold a checkpoint was tuned for.

    The operating point is a property of the model, so it is stored in the checkpoint
    rather than hard-coded at the call site. Checkpoints without the key fall back to
    ``default``.

    Parameters
    ----------
    model : str or torch.nn.Module
        Checkpoint path or module.
    default : float, optional
        Value for checkpoints that do not record one. Default is 0.5.

    Returns
    -------
    float
    """
    if isinstance(model, torch.nn.Module):
        return float(getattr(model, 'recommended_threshold', default))
    try:
        state = torch.load(model, map_location='cpu', weights_only=True)
    except (OSError, RuntimeError):
        return float(default)
    return float(state.get('recommended_threshold', default))


def polarity_rule(model):
    """Polarity rule a checkpoint decides with; see :class:`~contraction_net.polarity.PolarityRule`.

    Like the threshold, the rule's constants are a property of the model and stored in the
    checkpoint as ``'polarity_rule'``. Checkpoints without the key, such as every one
    trained before the rule existed, get the version-1 defaults. Only ``'q90'``
    checkpoints apply it.

    Parameters
    ----------
    model : str or torch.nn.Module
        Checkpoint path or module.

    Returns
    -------
    PolarityRule
    """
    if isinstance(model, torch.nn.Module):
        stored = getattr(model, 'polarity_rule', None)
    else:
        stored = torch.load(model, map_location='cpu', weights_only=True).get('polarity_rule')
    return stored if isinstance(stored, PolarityRule) else PolarityRule.from_checkpoint(stored)
