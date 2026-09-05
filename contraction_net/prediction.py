import os
import threading

import numpy as np
import torch
from scipy.signal import savgol_filter

from .contraction_net import ContractionNet, SymmetrizedContractionNet
from .utils import get_device

# select device
device = get_device()

# Loaded models, keyed by (path, mtime, device); prediction is called once per
# myofibril/domain, so re-reading the checkpoint each time dominated the runtime.
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()

#: Frames of padding added on either side before inference.
_RECEPTIVE_FIELD = 24

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

    state_dict = torch.load(model, map_location=device, weights_only=False)
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
        the trace, so a signal resting low must be inverted first. This is the
        default.
        ``'symmetric'`` centres on ``(P10 + P90) / 2``, which makes the conditioning exactly
        odd -- ``x -> -x`` maps level to ``-level`` and diff to ``-diff``, since
        ``P10(-x) = -P90(x)`` -- so a polarity-invariant model can be trained with sign-flip
        augmentation. Rest is then not distinguished by sign and must be inferred from the
        waveform.
        ``'raw'`` passes the trace through unchanged as a single channel; the pre-1.0
        recipe did no conditioning at all, and its comparison arm has to be fed the same
        way it was trained.

    Returns
    -------
    ndarray
        Array of shape ``(2, len(data))``: the referenced level, and its smoothed per-frame
        difference on the same scale. ``'raw'`` returns ``(1, len(data))``.

    Notes
    -----
    Both the reference and the scale are affine-equivariant, so the same trace in µm and in
    nm produces bit-identical output.
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


def predict_contractions(data, model, network=ContractionNet):
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

    Returns
    -------
    ndarray
        Array of shape ``(out_channels, len(data))`` with per-frame probabilities. Channel
        0 is the contraction state.

    Raises
    ------
    ValueError
        If the trace is empty or non-finite, or if ``model`` predates the 1.0 architecture.

    Notes
    -----
    The input is conditioned by :func:`prepare_robust_input` using the convention recorded
    in the checkpoint, so it is offset- and scale-invariant. Under the ``'q90'`` convention
    the reference is a high quantile rather than the median, which is not duty-robust; rest
    is then assumed to be the high side, and **signals whose resting state is the low side
    must be inverted before being passed in**. Models trained with the ``'symmetric'``
    convention carry no such requirement.
    """
    net = _load_model(model, network)

    data = np.asarray(data, dtype=np.float64).ravel()
    len_data = data.shape[0]
    if len_data == 0:
        raise ValueError('Cannot predict contractions for an empty time-series.')
    if not np.isfinite(data).all():
        raise ValueError('Time-series contains NaN or infinite values; fill gaps first.')

    prepared = prepare_robust_input(data, convention=getattr(net, 'input_convention', 'q90'))
    # a model trained on the level alone must not be handed the difference channel
    want = int(getattr(net, 'expected_in_channels', prepared.shape[0]))
    prepared = prepared[:want]

    # pad so the edge frames are inferred from context, not from a truncated window
    pad = max(_RECEPTIVE_FIELD, 2 - len_data)
    mode = 'reflect' if pad <= len_data - 1 else 'edge'
    prepared = np.pad(prepared, ((0, 0), (pad, pad)), mode=mode)

    tensor = torch.from_numpy(prepared).unsqueeze(0).to(device)
    with torch.no_grad():
        res = net(tensor)[0][0]
    res = res.detach().cpu().numpy()

    return res[:, pad:pad + len_data]


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
        state = torch.load(model, map_location='cpu', weights_only=False)
    except (OSError, RuntimeError):
        return float(default)
    return float(state.get('recommended_threshold', default))
