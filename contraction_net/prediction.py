import os
import threading

import numpy as np
import torch
from scipy.signal import savgol_filter

from .contraction_net import ContractionNet
from .utils import get_device

# select device
device = get_device()

# Loaded models, keyed by (path, mtime, device). Prediction is called once per
# myofibril/domain -- 56 times for a single packaged recording -- and re-reading a 7 MB
# checkpoint from disk each time dominated the runtime.
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()

#: Frames of context the convolutional stack needs on either side. Padding by at least this
#: much keeps the boundary from being computed against fabricated data.
_RECEPTIVE_FIELD = 24


#: Weight name unique to the architecture used before 1.0. Present only so that such a
#: checkpoint fails with a readable message instead of a state-dict key mismatch.
_PRE_1_0_MARKER = 'attention.in_proj_weight'


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
    if _PRE_1_0_MARKER in state_dict.get('state_dict', {}):
        raise ValueError(
            f'{model} was trained with the pre-1.0 ContractionNet architecture, which was '
            'removed in 1.0: it coupled its normalisation across the whole time axis and so '
            'lost contractions in recordings that spend most of their time contracting. '
            'Retrain with contraction_net.Trainer -- see the ContractionNet training '
            'tutorial -- or use the bundled model_ContractionNet.pt.')

    net = network(state_dict['n_filter'], in_channels=state_dict['in_channels'],
                  out_channels=state_dict['out_channels']).to(device)
    net.load_state_dict(state_dict['state_dict'])
    # Without this the module stays in training mode and its dropout keeps firing during
    # inference, making every call an unintended random ensemble member: repeated
    # predictions on one trace disagreed on several percent of frames and shifted the
    # detected cycle count, and therefore the beating rate, run to run.
    net.eval()

    if key is not None:
        with _CACHE_LOCK:
            _MODEL_CACHE[key] = net
    return net


def prepare_robust_input(data, diff_window=5):
    """Condition a 1D trace into the two-channel input used by :class:`ContractionNet`.

    Shared by training and inference so the two conventions cannot drift apart.

    Parameters
    ----------
    data : ndarray
        1D time-series, finite.
    diff_window : int, optional
        Smoothing window (frames) for the difference channel. Default is 5.

    Returns
    -------
    ndarray
        Array of shape ``(2, len(data))``: the level referenced to its 90th percentile and
        scaled by the 10-90 spread, and the smoothed per-frame difference on the same scale.

    Notes
    -----
    Both the reference and the scale are invariant to affine changes of the input, so the
    same trace in µm and in nm produces bit-identical output. See
    :func:`predict_contractions` for why the reference is a high quantile rather than the
    median.
    """
    x = np.asarray(data, dtype=np.float64).ravel()
    rest = np.percentile(x, 90)
    scale = float(np.percentile(x, 90) - np.percentile(x, 10))
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
    The input is conditioned by :func:`prepare_robust_input`: referenced to a high quantile,
    scaled by the 10-90 spread, and paired with a per-frame difference channel. This is
    genuinely offset- and scale-invariant, so the same trace in µm and in nm gives
    bit-identical output.

    The reference is the 90th percentile rather than the median because the median is *not*
    duty-robust: in a trace that spends 90% of its time contracting the median sits inside
    the contraction, so centring on it would feed the network a duty-dependent offset and
    reintroduce the very bias the architecture removes. Rest is the high side of a
    sarcomere-length trace, so a high quantile stays near rest up to duty ~0.9. **Signals
    whose resting state is the low side must be inverted before being passed in.**
    """
    net = _load_model(model, network)

    data = np.asarray(data, dtype=np.float64).ravel()
    len_data = data.shape[0]
    if len_data == 0:
        raise ValueError('Cannot predict contractions for an empty time-series.')
    if not np.isfinite(data).all():
        raise ValueError('Time-series contains NaN or infinite values; fill gaps first.')

    prepared = prepare_robust_input(data)

    # Pad by the receptive field so the first and last frames are inferred from real
    # context, and so that normalisation always has more than one sample to work with.
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

    The right operating point is a property of the model, not of the caller, so it is
    stored in the checkpoint and read back rather than hard-coded at the call site: a
    threshold carried over from a differently-trained model is not meaningful. A sweep puts
    the best trade-off for the bundled model at 0.5 -- stress-set IoU 0.759 with 15.8% false
    positives on quiescent traces, against 0.742 and 27.2% at 0.3. Checkpoints without the
    key fall back to ``default``.

    Parameters
    ----------
    model : str or torch.nn.Module
        Checkpoint path or module.
    default : float, optional
        Value for checkpoints that do not record one. Default is 0.3.

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
