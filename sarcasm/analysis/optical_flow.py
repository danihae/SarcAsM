"""Dense image optical flow as a motion predictor for the sarcomere-vector tracker.

The 2D tracker (:mod:`sarcasm.analysis.sarcomere_tracking`) matches each query
point to a detection inside a gate around its *predicted* position. At high frame
rates a sarcomere moves a fraction of a pixel per frame and holding the last
position is the best prediction. When the frame rate is coarse relative to the
motion — one contraction moves a sarcomere a sizeable fraction of its length
between consecutive frames — the hold-position prediction falls outside the gate
and identities fragment. This module supplies the displacement field between consecutive raw
frames so the tracker can carry every query point along with the tissue before
gating.

Design notes (validated on high-frame-rate movies, their temporal subsamples, and
coarse-frame-rate recordings):

* The flow is computed on the **raw image**, not on segmentation masks — mask
  flicker turns into fabricated motion.
* OpenCV's DIS optical flow (dense inverse search) is used: equal in accuracy to
  Farneback and sparse pyramidal Lucas–Kanade at the detections, ~3× faster than
  Farneback; the cost scales with the number of pixels.
* The field is block-averaged to 1/``downsample`` resolution (the flow is smooth
  at the scale of a sarcomere) and sampled bilinearly at the query points, so only
  one reduced field is held in memory at a time.
* The flow's *perpendicular* component (along the striations) is far noisier than
  the *along-axis* component (aperture problem): at high frame rates its error
  exceeds the true lateral motion by more than an order of magnitude. The tracker
  therefore applies the along-axis component
  always and the perpendicular one only where it is large relative to the
  sarcomere length (see ``_PREDICT_PERP_KEEP_FRAC`` in the tracker).

Nothing is persisted; the predictor is rebuilt per tracking run.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy import ndimage


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "The image-flow motion predictor needs OpenCV; install 'opencv-python-headless'."
        ) from e
    return cv2


def intensity_range(img: np.ndarray, percentiles: Tuple[float, float] = (0.5, 99.9)) -> Tuple[float, float]:
    """Intensity window ``(lo, hi)`` for :func:`frame_to_uint8`, from one frame's percentiles.

    Parameters
    ----------
    img : np.ndarray
        A 2D frame.
    percentiles : (float, float), optional
        Lower / upper percentile. Default is (0.5, 99.9).

    Returns
    -------
    tuple of float
        ``(lo, hi)`` with ``hi > lo`` guaranteed.
    """
    lo, hi = np.percentile(np.asarray(img, dtype=np.float32), percentiles)
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def frame_to_uint8(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip a frame to ``[lo, hi]`` and rescale to uint8 for the flow estimator.

    Parameters
    ----------
    img : np.ndarray
        A 2D frame of any numeric dtype.
    lo, hi : float
        Intensity window; use the same window for every frame of a movie so the
        flow does not see a brightness step.

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint8.
    """
    x = (np.asarray(img, dtype=np.float32) - lo) / (hi - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def dis_flow(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    """Dense displacement field from frame ``a`` to frame ``b`` (DIS optical flow).

    Parameters
    ----------
    a_u8, b_u8 : np.ndarray
        ``(H, W)`` uint8 frames.

    Returns
    -------
    np.ndarray
        ``(H, W, 2)`` float32 displacement in px as ``[dy, dx]`` (numpy row/col
        order): a feature at ``(y, x)`` in ``a`` is at ``(y + dy, x + dx)`` in ``b``.
    """
    cv2 = _cv2()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(1)
    flow_xy = dis.calc(np.ascontiguousarray(a_u8), np.ascontiguousarray(b_u8), None)
    return np.ascontiguousarray(flow_xy[..., ::-1]).astype(np.float32, copy=False)


def block_mean(flow: np.ndarray, downsample: int) -> np.ndarray:
    """Block-average a ``(H, W, 2)`` field to ``(H // ds, W // ds, 2)``.

    Parameters
    ----------
    flow : np.ndarray
        ``(H, W, 2)`` field.
    downsample : int
        Block edge in px; 1 returns the field unchanged.

    Returns
    -------
    np.ndarray
        float32 block means (trailing rows/cols that do not fill a block are dropped).
    """
    ds = int(downsample)
    if ds <= 1:
        return np.asarray(flow, dtype=np.float32)
    H, W = flow.shape[:2]
    h, w = H // ds, W // ds
    return flow[:h * ds, :w * ds].reshape(h, ds, w, ds, 2).mean(axis=(1, 3)).astype(np.float32)


def sample_flow(flow_ds: np.ndarray, yx: np.ndarray, downsample: int) -> np.ndarray:
    """Bilinearly sample a block-averaged field at full-resolution positions.

    Parameters
    ----------
    flow_ds : np.ndarray
        ``(h, w, 2)`` field from :func:`block_mean`.
    yx : np.ndarray
        ``(Q, 2)`` positions in full-resolution px (row, col).
    downsample : int
        The block edge used to build ``flow_ds``.

    Returns
    -------
    np.ndarray
        ``(Q, 2)`` float32 displacement ``[dy, dx]`` at each position.
    """
    q = np.asarray(yx, dtype=np.float64).reshape(-1, 2)
    if q.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    ds = float(downsample)
    # block (i, j) is centred at ((i + 0.5) * ds - 0.5) in full-resolution px
    coords = ((q - (ds - 1.0) / 2.0) / ds).T
    out = np.column_stack([
        ndimage.map_coordinates(flow_ds[..., k], coords, order=1, mode='nearest') for k in range(2)
    ])
    return out.astype(np.float32)


class ImageFlowPredictor:
    """Per-step displacement predictor for :func:`~sarcasm.analysis.sarcomere_tracking.track_sarcomere_vectors`.

    Calling ``predictor(k, yx)`` returns the displacement from tracking step ``k``
    to ``k + 1`` at the positions ``yx`` (``(Q, 2)`` px). The flow between the two
    raw frames is computed on first use of a step and only the latest field (and
    the two uint8 frames) are kept in memory.

    Parameters
    ----------
    read_frame : callable
        ``read_frame(k) -> (H, W)`` array: the raw frame of tracking step ``k``
        (the caller maps step indices to movie frames).
    downsample : int, optional
        Block edge for :func:`block_mean`. Default is 4.
    percentiles : (float, float), optional
        Intensity window percentiles, taken from the first frame read. Default is
        (0.5, 99.9).
    """

    def __init__(self, read_frame: Callable[[int], np.ndarray], downsample: int = 4,
                 percentiles: Tuple[float, float] = (0.5, 99.9)) -> None:
        self._read = read_frame
        self.downsample = int(downsample)
        self._percentiles = percentiles
        self._window: Optional[Tuple[float, float]] = None
        self._u8: dict = {}
        self._field: Tuple[Optional[int], Optional[np.ndarray]] = (None, None)

    def _frame_u8(self, k: int) -> np.ndarray:
        if k not in self._u8:
            img = np.asarray(self._read(k))
            if img.ndim == 3 and img.shape[0] == 1:   # a one-frame stack slice
                img = img[0]
            if img.ndim != 2:
                raise ValueError(f"read_frame({k}) must return a 2D frame, got shape {img.shape}.")
            if self._window is None:
                self._window = intensity_range(img, self._percentiles)
            self._u8 = {j: f for j, f in self._u8.items() if j >= k - 1}  # keep at most the previous frame
            self._u8[k] = frame_to_uint8(img, *self._window)
        return self._u8[k]

    def field(self, k: int) -> np.ndarray:
        """Block-averaged displacement field for step ``k`` (frame ``k`` → ``k + 1``)."""
        if self._field[0] != k:
            a, b = self._frame_u8(k), self._frame_u8(k + 1)
            self._field = (k, block_mean(dis_flow(a, b), self.downsample))
        return self._field[1]

    def __call__(self, k: int, yx: np.ndarray) -> np.ndarray:
        return sample_flow(self.field(k), yx, self.downsample)
