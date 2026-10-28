"""Edge padding shared by inference and training, so both see the same trace edges."""
import numpy as np

__all__ = ['INFERENCE_PAD', 'pad_for_inference']

#: Frames added on each side of a conditioned trace before the network reads it. This is
#: not the receptive field (513 frames); it only keeps the first and last frames from being
#: read off a window that ends abruptly.
INFERENCE_PAD = 24


def pad_for_inference(prepared, pad=INFERENCE_PAD):
    """Pad a conditioned trace on both sides along its last (time) axis.

    Parameters
    ----------
    prepared : ndarray
        Conditioned input of shape ``(channels, n)``.
    pad : int, optional
        Frames to add on each side. Default is :data:`INFERENCE_PAD`.

    Returns
    -------
    ndarray
        The padded array, of shape ``(channels, n + 2 * pad)``.
    int
        The number of frames added on each side, to crop the prediction back.

    Notes
    -----
    The padding reflects the trace when it is longer than ``pad`` frames and repeats the
    edge values otherwise, as ``predict_contractions`` always has.
    """
    prepared = np.asarray(prepared)
    n = prepared.shape[-1]
    mode = 'reflect' if pad <= n - 1 else 'edge'
    widths = [(0, 0)] * (prepared.ndim - 1) + [(pad, pad)]
    return np.pad(prepared, widths, mode=mode), pad
