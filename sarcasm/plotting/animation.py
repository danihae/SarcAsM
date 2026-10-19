# -*- coding: utf-8 -*-
"""Movies of SarcAsM results: raw movie and Z-band masks, live sarcomere-vector colormaps, and
composite results movies with grouped time series and a time cursor.

The public entry points are the static methods of :class:`Animations`. They build a figure with
the usual :class:`~sarcasm.plotting.plots.Plots` functions, update a small set of artists frame by
frame and write an ``.mp4`` (H.264 via ffmpeg) or ``.gif`` (Pillow). ffmpeg is looked up on the
``PATH`` first and then in the optional ``imageio-ffmpeg`` package.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.axes import Axes
from matplotlib_scalebar.scalebar import ScaleBar

from sarcasm.plotting.plot_utils import PlotUtils
from sarcasm.plotting.plots import Plots
from sarcasm.structure import SarcAsM

FramesSpec = Union[None, Tuple[float, float], Sequence[int], np.ndarray]


def _movie_frame(sarc_obj: SarcAsM, position: int) -> int:
    """Movie frame number at ``position`` of the tracked frame list. The renderers are driven by these
    positions (see :meth:`Animations.resolve_frames`); a tracking that did not start at frame 0 shifts them
    against the movie. Without tracks the positions are the movie frames."""
    tracked = sarc_obj.data.get('params.track_sarcomere_vectors.frames') if 'motion.tracks.n' in sarc_obj.data else None
    return int(position) if tracked is None else int(np.asarray(tracked).ravel()[int(position)])

#: Shading of ContractionNet contraction periods in the time-series panels.
CONTR_COLOR = 'lavender'


# --------------------------------------------------------------------------- frame renderers
class _ImageLayer:
    """Raw movie or Z-band mask of one frame, ``imshow`` updated in place.

    The display range of the raw image is fixed on the first frame so the brightness does not
    flicker; the mask is drawn white on black (``style='mask'``) or as dark bands on a light
    background (``style='bands'``, used under the ΔSL colormap so that near-zero values stay visible).
    """

    def __init__(self, sarc_obj: SarcAsM, frames: np.ndarray, layer: str = 'image', style: str = 'mask',
                 clip: Tuple[float, float] = (1, 99.5), scalebar: bool = True, alpha: float = 1.0):
        self.sarc, self.layer, self.style, self.clip, self.scalebar, self.alpha = sarc_obj, layer, style, clip, scalebar, alpha
        self.frames = np.asarray(frames, dtype=int)
        self._cache: Dict[int, np.ndarray] = {}
        self.im = None
        self.lo = self.hi = None

    def read(self, frame: int) -> np.ndarray:
        if frame not in self._cache:
            movie_frame = _movie_frame(self.sarc, frame)
            if self.layer == 'image':
                self._cache[frame] = np.asarray(self.sarc.read_imgs(frames=movie_frame), dtype=float)
            else:
                name = 'zbands_fast_movie' if self.sarc._mask_exists('zbands_fast_movie') else 'zbands'
                self._cache[frame] = np.asarray(self.sarc._read_mask(name, frames=movie_frame), dtype=float)
        return self._cache[frame]

    def preload(self, max_bytes: float = 1.5e9) -> None:
        """Read all frames once when they fit comfortably into memory."""
        h, w = self.sarc.metadata.size[-2:]
        if self.frames.size * h * w * 8 > max_bytes:
            return
        movie_frames = [_movie_frame(self.sarc, f) for f in self.frames]
        if self.layer == 'image':
            arr = np.asarray(self.sarc.read_imgs(frames=movie_frames), dtype=float)
        else:
            name = 'zbands_fast_movie' if self.sarc._mask_exists('zbands_fast_movie') else 'zbands'
            arr = np.asarray(self.sarc._read_mask(name, frames=movie_frames), dtype=float)
        arr = arr.reshape(self.frames.size, h, w)
        self._cache = {int(f): arr[i] for i, f in enumerate(self.frames)}

    def _render(self, frame: int) -> np.ndarray:
        img = self.read(frame)
        if self.layer == 'image':
            return np.clip(img, self.lo, self.hi)
        if self.style == 'bands':
            return np.ma.masked_less(img, 0.5)
        return img

    def setup(self, ax: Axes, frame: int) -> None:
        img = self.read(frame)
        if self.layer == 'image':
            self.lo, self.hi = np.percentile(img, self.clip)
            self.im = ax.imshow(self._render(frame), cmap='gray', vmin=self.lo, vmax=self.hi, alpha=self.alpha)
        elif self.style == 'bands':
            ax.set_facecolor('0.8')
            self.im = ax.imshow(self._render(frame), cmap='Greys', vmin=0, vmax=1.6, alpha=self.alpha)
        else:
            self.im = ax.imshow(self._render(frame), cmap='gray', vmin=0, vmax=1, alpha=self.alpha)
        h, w = img.shape
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        if self.scalebar:
            ax.add_artist(ScaleBar(self.sarc.metadata.pixelsize, units='µm', frameon=False, color='w', sep=1,
                                   width_fraction=0.015, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 2}))

    def update(self, frame: int) -> None:
        self.im.set_data(self._render(frame))


class _TrackVectorLayer:
    """Every tracked sarcomere of a frame as its Z-band-to-Z-band line, coloured by ``color_by``
    (see :meth:`Plots.plot_track_vectors`); ``highlight`` outlines selected tracks."""

    def __init__(self, sarc_obj: SarcAsM, color_by: str = 'slen', cmap: Optional[str] = None,
                 clim: Optional[Tuple[float, float]] = None, linewidth: Optional[float] = None,
                 only_observed: bool = True, colorbar: bool = True,
                 highlight: Optional[Sequence[np.ndarray]] = None, kinematics: Optional[dict] = None):
        self.sarc, self.color_by, self.cmap, self.clim = sarc_obj, color_by, cmap, clim
        self.linewidth, self.only_observed, self.colorbar = linewidth, only_observed, colorbar
        self.highlight = highlight or []
        self.kin = kinematics
        self.values = None
        self.lc = None

    def setup(self, ax: Axes, frame: int) -> None:
        self.values = Plots.track_color_values(self.sarc, self.color_by, self.kin)
        self.lc = Plots.plot_track_vectors(ax, self.sarc, _movie_frame(self.sarc, frame), color_by=self.color_by,
                                           cmap=self.cmap, clim=self.clim, linewidth=self.linewidth,
                                           only_observed=self.only_observed, scalebar=False, colorbar=False,
                                           kinematics=self.kin)
        if self.colorbar and self.color_by != 'group':
            bb = ax.get_position()
            cax = ax.figure.add_axes([bb.x1 + 0.006, bb.y0 + 0.1 * (bb.y1 - bb.y0), 0.008, 0.8 * (bb.y1 - bb.y0)])
            cb = ax.figure.colorbar(self.lc, cax=cax)
            cb.set_label(Plots.TRACK_COLOR_DEFAULTS[self.color_by][2], fontsize=PlotUtils.fontsize - 2, labelpad=1)
            cb.ax.tick_params(labelsize=PlotUtils.fontsize - 3, length=2)
        n_tracks = int(self.sarc.data['motion.tracks.n'])
        pos = np.asarray(self.sarc.data['motion.tracks.positions_px'], dtype=float).reshape(n_tracks, -1, 2)[:, frame]
        for members in self.highlight:
            members = np.asarray(members, dtype=int)
            ax.plot(pos[members, 1], pos[members, 0], '-', color='w', lw=1.8, alpha=0.95, zorder=5)
            ax.plot(pos[members, 1], pos[members, 0], '-', color='k', lw=0.8, alpha=0.95, zorder=6)

    def update(self, frame: int) -> None:
        movie_frame = _movie_frame(self.sarc, frame)
        mask = Plots.track_vector_mask(self.sarc, movie_frame, only_observed=self.only_observed)
        vals = self.values[:, frame] if self.values.ndim == 2 else self.values
        if self.color_by == 'group':
            mask &= vals >= 0
        mask &= np.isfinite(vals)
        self.lc.set_segments(Plots.track_vector_segments(self.sarc, movie_frame, mask))
        if self.color_by == 'group':
            n_groups = max(int(self.sarc.data.get('motion.groups.n', 0)), 1)
            self.lc.set_colors(plt.get_cmap(self.cmap or 'gist_rainbow')((vals[mask].astype(int) % n_groups) / n_groups))
        else:
            self.lc.set_array(vals[mask])


class _TimeCursor:
    """Dashed vertical line at the current time on time-series axes, plus a ``t = … s`` label."""

    def __init__(self, time: np.ndarray, axes: Sequence[Axes], label_text=None):
        self.time, self.axes, self.label = np.asarray(time, dtype=float), list(axes), label_text
        self.lines = []

    def setup(self, ax: Axes, frame: int) -> None:
        self.lines = [a.axvline(self.time[frame], color='k', ls='--', lw=1.0, zorder=6) for a in self.axes]
        if self.label is not None:
            self.label.set_text(f't = {self.time[frame]:.2f} s')

    def update(self, frame: int) -> None:
        for line in self.lines:
            line.set_xdata([self.time[frame], self.time[frame]])
        if self.label is not None:
            self.label.set_text(f't = {self.time[frame]:.2f} s')


# --------------------------------------------------------------------------- public API
class Animations:
    """Movies of SarcAsM results (static methods, in the style of :class:`Plots`)."""

    # ----------------------------------------------------------------- writer
    @staticmethod
    def find_ffmpeg() -> Optional[str]:
        """Path of a working ffmpeg executable: ``$PATH`` first, then the ``imageio-ffmpeg`` package,
        else None. A binary that cannot run ``ffmpeg -version`` (e.g. broken shared libraries) is skipped."""
        candidates = [shutil.which('ffmpeg')]
        try:
            import imageio_ffmpeg
            candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
        except ImportError:
            pass
        for exe in candidates:
            if not exe:
                continue
            try:
                if subprocess.run([exe, '-version'], capture_output=True, timeout=20).returncode == 0:
                    return exe
            except (OSError, subprocess.SubprocessError):
                continue
        return None

    @staticmethod
    @contextmanager
    def movie_writer(fig, out: str, fps: float, dpi: int = 160, crf: int = 18, codec: str = 'libx264'
                     ) -> Iterator[callable]:
        """
        Context manager yielding a ``grab()`` callable that appends the current state of ``fig``.

        ``out`` ending in ``.mp4`` uses ffmpeg (H.264, ``yuv420p``; the figure size is padded to even
        pixel dimensions), any other extension uses the Pillow GIF writer.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to record.
        out : str
            Output path (``.mp4`` or ``.gif``).
        fps : float
            Playback frame rate.
        dpi : int, optional
            Resolution. Default is 160.
        crf : int, optional
            H.264 quality (lower = better, 18 is visually lossless). Default is 18.
        codec : str, optional
            ffmpeg video codec. Default is 'libx264'.
        """
        out = os.path.expanduser(out)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        if out.lower().endswith('.mp4'):
            exe = Animations.find_ffmpeg()
            if exe is None:
                raise RuntimeError('No ffmpeg found. Install ffmpeg or `pip install imageio-ffmpeg`, '
                                   'or write a .gif instead.')
            plt.rcParams['animation.ffmpeg_path'] = exe
            w, h = fig.get_size_inches()
            fig.set_size_inches(np.ceil(w * dpi / 2) * 2 / dpi, np.ceil(h * dpi / 2) * 2 / dpi)
            writer = FFMpegWriter(fps=fps, codec=codec, bitrate=-1,
                                  extra_args=['-pix_fmt', 'yuv420p', '-crf', str(crf), '-preset', 'medium'])
        else:
            writer = PillowWriter(fps=fps)
        with writer.saving(fig, out, dpi=dpi):
            yield writer.grab_frame

    # ----------------------------------------------------------------- frame selection
    @staticmethod
    def resolve_frames(sarc_obj: SarcAsM, frames: FramesSpec = None) -> np.ndarray:
        """
        Frame indices (positions in the tracked frame list) for a frames specification.

        Parameters
        ----------
        sarc_obj : SarcAsM
            Analysed object (its frametime converts seconds to frames).
        frames : None, (start_s, stop_s) or sequence of int
            ``None`` = all tracked frames
            a tuple = time window in seconds
            a sequence = explicit indices.
        """
        n = int(np.asarray(sarc_obj.data['motion.tracks.observed']).reshape(int(sarc_obj.data['motion.tracks.n']), -1).shape[1])
        if frames is None:
            return np.arange(n)
        if isinstance(frames, tuple) and len(frames) == 2:
            ft = float(sarc_obj.metadata.frametime)
            a, b = int(round(frames[0] / ft)), int(round(frames[1] / ft))
            return np.arange(max(a, 0), min(b, n - 1) + 1)
        idx = np.asarray(frames, dtype=int)
        return idx[(idx >= 0) & (idx < n)]

    @staticmethod
    def playback_fps(sarc_obj: SarcAsM, fps: Optional[float] = None, slow_motion: float = 1.0) -> float:
        """Frame rate of the file: ``fps`` if given, else real time divided by ``slow_motion``."""
        base = fps if fps is not None else 1.0 / float(sarc_obj.metadata.frametime)
        return float(base) / float(slow_motion)

    @staticmethod
    def _group_members(sarc_obj: SarcAsM, kind: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Group ids of all tracks for ``kind`` (per-kind mirror or the current grouping) and the
        order along the chain for the chain levels."""
        key = f'motion.groups.{kind}.track_group_id'
        if key in sarc_obj.data:
            gid = np.asarray(sarc_obj.data[key]).reshape(-1).astype(int)
            order = sarc_obj.data.get(f'motion.groups.{kind}.track_group_order')
        elif sarc_obj.data.get('motion.groups.kind') == kind:
            gid = np.asarray(sarc_obj.data['motion.tracks.group_id']).reshape(-1).astype(int)
            order = sarc_obj.data.get('motion.tracks.group_order')
        else:
            raise ValueError(f"no '{kind}' grouping found; run analyze_track_motion(by='{kind}') first")
        return gid, (np.asarray(order).reshape(-1) if order is not None else None)

    @staticmethod
    def _window_members(sarc_obj: SarcAsM, frames: np.ndarray, min_coverage: float) -> np.ndarray:
        """Boolean mask of the tracks observed (with a finite length) in at least ``min_coverage`` of ``frames``."""
        n_tracks = int(sarc_obj.data['motion.tracks.n'])
        obs = np.asarray(sarc_obj.data['motion.tracks.observed']).reshape(n_tracks, -1)[:, frames].astype(bool)
        slen = np.asarray(sarc_obj.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)[:, frames]
        return (obs & np.isfinite(slen)).mean(axis=1) >= min_coverage

    @staticmethod
    def auto_group(sarc_obj: SarcAsM, kind: str, frames: np.ndarray, min_members: int = 8,
                   size_range: Optional[Tuple[int, int]] = None, min_coverage: float = 0.8,
                   criterion: str = 'size') -> int:
        """
        Pick a representative group of ``kind`` for a movie window: members must be observed in at
        least ``min_coverage`` of ``frames``; among the groups with at least ``min_members`` such
        members (inside ``size_range`` if given) the one with the lowest frame-to-frame jitter wins,
        with larger groups (``criterion='size'``) or, for chains, the longest end-to-end span in µm
        (``criterion='extent'``) preferred first.

        Returns
        -------
        int
            Group id.
        """
        gid, order = Animations._group_members(sarc_obj, kind)
        n_tracks = gid.size
        slen = np.asarray(sarc_obj.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)[:, frames]
        pos0 = np.asarray(sarc_obj.data['motion.tracks.positions_um'], dtype=float).reshape(n_tracks, -1, 2)[:, frames[0]]
        full = Animations._window_members(sarc_obj, frames, min_coverage)
        best = None
        for g in np.unique(gid[gid >= 0]):
            mm = np.flatnonzero((gid == g) & full)
            if mm.size < min_members:
                continue
            jitter = float(np.nanmean(np.nanstd(np.diff(slen[mm], axis=1), axis=1)))
            size_key = 0 if size_range is None or size_range[0] <= mm.size <= size_range[1] else 1
            if criterion == 'extent':
                mo = mm[np.argsort(order[mm])] if order is not None else mm
                extent = float(np.hypot(*(pos0[mo[-1]] - pos0[mo[0]])))
                key = (size_key, -round(extent, 1), jitter)
            else:
                key = (size_key, -mm.size if size_range is None else 0, jitter)
            if best is None or key < best[0]:
                best = (key, int(g))
        if best is None:
            raise ValueError(f"no '{kind}' group with >= {min_members} members observed in every frame")
        return best[1]

    @staticmethod
    def _parse_group(sarc_obj: SarcAsM, spec: str, frames: np.ndarray,
                     min_coverage: float = 0.8) -> Tuple[str, int, np.ndarray]:
        """``'pool'``, ``'mband:auto'`` or ``'myofibril:8'`` -> (kind, group id, member track indices sorted
        along the group). Only members observed in at least ``min_coverage`` of ``frames`` are kept;
        for the pool these are all such tracks."""
        kind, _, which = spec.partition(':')
        n_tracks = int(sarc_obj.data['motion.tracks.n'])
        full = Animations._window_members(sarc_obj, frames, min_coverage)
        if kind == 'pool':
            return 'pool', 0, np.flatnonzero(full)
        gid, order = Animations._group_members(sarc_obj, kind)
        if which in ('', 'auto', 'size', 'extent'):
            g = Animations.auto_group(sarc_obj, kind, frames, criterion='extent' if which == 'extent' else 'size')
        else:
            g = int(which)
        members = np.flatnonzero((gid == g) & full)
        if members.size == 0:
            raise ValueError(f"group {g} of '{kind}' has no member with coverage >= {min_coverage:.0%} in the window")
        if order is not None and kind in ('myofibril', 'loi'):
            members = members[np.argsort(order[members])]
        else:  # lateral position along the M-band
            pos0 = np.asarray(sarc_obj.data['motion.tracks.positions_px'], dtype=float).reshape(n_tracks, -1, 2)[:, 0]
            members = members[np.argsort(pos0[members, 0])]
        return kind, g, members

    # ----------------------------------------------------------------- time-series panel
    @staticmethod
    def _draw_group_overlay(ax: Axes, sarc_obj: SarcAsM, kind: str, members: np.ndarray, frames: np.ndarray,
                            quantity: str, kinematics: dict, percentiles: Tuple[float, ...] = (5, 25, 75, 95),
                            n_faint: int = 60, seed: int = 0, y_lim: Optional[Tuple[float, float]] = None,
                            title: Optional[str] = None) -> None:
        """Time series of a group over ``frames``: faint member traces (for the pool a random subset
        plus percentile bands), the group mean (black), the cell average (dashed, non-pool groups)
        and the ContractionNet contraction periods."""
        ft = float(sarc_obj.metadata.frametime)
        n_tracks = int(sarc_obj.data['motion.tracks.n'])
        values = kinematics['slen'] if quantity == 'slen' else kinematics['delta_slen']
        values = np.asarray(values, dtype=float).reshape(n_tracks, -1)[:, frames]
        obs = np.asarray(sarc_obj.data['motion.tracks.observed']).reshape(n_tracks, -1)[:, frames].astype(bool)
        values = np.where(obs, values, np.nan)
        t = frames * ft
        full = members
        if 'motion.pool.contr' in sarc_obj.data:
            contr = np.asarray(sarc_obj.data['motion.pool.contr']).reshape(-1).astype(bool)[frames]
            edges = np.flatnonzero(np.diff(np.r_[0, contr.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                ax.axvspan(t[a], t[min(b, t.size - 1)], color=CONTR_COLOR, lw=0, zorder=0)
        if kind == 'pool':
            rng = np.random.default_rng(seed)
            show = rng.choice(full, min(n_faint, full.size), replace=False) if full.size else full
            for k in show:
                ax.plot(t, values[k], lw=0.3, color='0.2', alpha=0.18, zorder=1)
            if full.size and len(percentiles) == 4:
                q = np.nanpercentile(values[full], percentiles, axis=0)
                ax.fill_between(t, q[0], q[3], color='#9fb6d9', alpha=0.45, lw=0, zorder=2,
                                label=f'{percentiles[0]:g}–{percentiles[3]:g}%')
                ax.fill_between(t, q[1], q[2], color='#4c78b4', alpha=0.55, lw=0, zorder=2.2,
                                label=f'{percentiles[1]:g}–{percentiles[2]:g}%')
            pool_key = 'motion.pool.slen' if quantity == 'slen' else None
            if pool_key in sarc_obj.data:
                mean = np.asarray(sarc_obj.data[pool_key], dtype=float).reshape(-1)[frames]
            else:
                mean = np.nanmean(values[members], axis=0)
            ax.plot(t, mean, color='k', lw=1.3, zorder=3, label='cell average')
            ax.legend(loc='upper right', fontsize=PlotUtils.fontsize - 3, frameon=False, ncol=3,
                      handlelength=1.2, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.1)
        else:
            cmap = plt.get_cmap('turbo')
            thin = members.size > 20
            for j, k in enumerate(members):
                ax.plot(t, values[k], lw=0.3 if thin else 0.45, color=cmap(j / max(members.size - 1, 1)),
                        alpha=0.3 if thin else 0.4, zorder=2)
            if 'motion.pool.slen' in sarc_obj.data and quantity == 'slen':
                cell = np.asarray(sarc_obj.data['motion.pool.slen'], dtype=float).reshape(-1)[frames]
            else:
                cov_all = np.flatnonzero(obs.mean(axis=1) >= 0.5)
                cell = np.nanmean(values[cov_all], axis=0)
            ax.plot(t, cell, color='0.45', lw=0.7, ls='--', zorder=2.5, label='cell average')
            ax.plot(t, np.nanmean(values[members], axis=0), color='k', lw=1.3, zorder=3, label='group mean')
            ax.legend(loc='upper right', fontsize=PlotUtils.fontsize - 3, frameon=False, ncol=2,
                      handlelength=1.4, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.1)
        ax.set_xlim(t[0], t[-1])
        if y_lim is not None:
            ax.set_ylim(*y_lim)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('SL [µm]' if quantity == 'slen' else 'ΔSL [µm]')
        if title:
            ax.set_title(title, fontsize=PlotUtils.fontsize, fontweight='bold')

    # ----------------------------------------------------------------- animators
    @staticmethod
    def _run(fig, renderers: Sequence, frames: np.ndarray, out: str, fps: float, dpi: int, progress: bool,
             snapshot: bool = False) -> str:
        with Animations.movie_writer(fig, out, fps=fps, dpi=dpi) as grab:
            for i, f in enumerate(frames):
                for r in renderers:
                    r.update(int(f))
                grab()
                if snapshot and i == 0:
                    fig.savefig(os.path.splitext(os.path.expanduser(out))[0] + '_frame0.png', dpi=dpi)
                if progress and i % 100 == 0:
                    print(f'frame {i}/{frames.size}', flush=True)
        plt.close(fig)
        return out

    @staticmethod
    def animate_image(sarc_obj: SarcAsM, out: str, frames: FramesSpec = None,
                      layers: Sequence[str] = ('image', 'zbands'), titles: Optional[Sequence[str]] = None,
                      fps: Optional[float] = None, slow_motion: float = 1.0, dpi: int = 160,
                      panel_width: float = 4.5, progress: bool = False) -> str:
        """
        Side-by-side movie of the raw recording and/or the Z-band masks.

        Parameters
        ----------
        sarc_obj : SarcAsM
            Object with the movie (and, for ``'zbands'``, the Z-band masks).
        out : str
            Output path (``.mp4`` or ``.gif``).
        frames : None, (start_s, stop_s) or sequence of int, optional
            Frames to render; default all tracked frames (or all frames of the stack when no tracks).
        layers : sequence of {'image', 'zbands'}, optional
            Panels from left to right. Default is ('image', 'zbands').
        titles : sequence of str, optional
            Panel titles; default 'Confocal movie' / 'Z-bands (3D U-Net)'.
        fps : float, optional
            Playback rate; default real time.
        slow_motion : float, optional
            Slow-down factor applied to the playback rate. Default is 1.
        dpi : int, optional
            Resolution. Default is 160.
        panel_width : float, optional
            Width of each panel in inches. Default is 4.5.
        progress : bool, optional
            Print progress every 100 frames. Default is False.

        Returns
        -------
        str
            The output path.
        """
        if 'motion.tracks.n' in sarc_obj.data:
            frames = Animations.resolve_frames(sarc_obj, frames)
        else:
            n = int(sarc_obj.metadata.n_stack)
            frames = np.arange(n) if frames is None else np.asarray(frames, dtype=int)
        h, w = sarc_obj.metadata.size[-2:]
        ph = panel_width * h / w
        fig = plt.figure(figsize=(0.3 + len(layers) * (panel_width + 0.3), ph + 0.6), dpi=dpi)
        titles = titles or [{'image': 'Confocal movie', 'zbands': 'Z-bands (3D U-Net)'}[layer] for layer in layers]
        renderers = []
        for i, (layer, title) in enumerate(zip(layers, titles)):
            ax = fig.add_axes([(0.3 + i * (panel_width + 0.3)) / fig.get_figwidth(), 0.1 / (ph + 0.6),
                               panel_width / fig.get_figwidth(), ph / (ph + 0.6)])
            r = _ImageLayer(sarc_obj, frames, layer=layer)
            r.preload()
            r.setup(ax, int(frames[0]))
            ax.set_title(title, fontsize=PlotUtils.fontsize, fontweight='bold')
            renderers.append(r)
        return Animations._run(fig, renderers, frames, out, Animations.playback_fps(sarc_obj, fps, slow_motion), dpi, progress)

    @staticmethod
    def animate_track_vectors(sarc_obj: SarcAsM, out: str, color_by: str = 'slen', frames: FramesSpec = None,
                              background: str = 'image', cmap: Optional[str] = None,
                              clim: Optional[Tuple[float, float]] = None, linewidth: Optional[float] = None,
                              highlight_groups: Optional[Dict[str, int]] = None, title: Optional[str] = None,
                              fps: Optional[float] = None, slow_motion: float = 1.0, dpi: int = 160,
                              panel_width: float = 6.0, progress: bool = False) -> str:
        """
        One-panel movie of the live sarcomere-vector colormap (:meth:`Plots.plot_track_vectors`
        updated every frame) over the raw movie or the Z-band mask.

        Parameters
        ----------
        sarc_obj : SarcAsM
            Object with tracks.
        out : str
            Output path (``.mp4`` or ``.gif``).
        color_by : {'slen', 'delta_slen', 'vel', 'coverage', 'group'}, optional
            Colour quantity. Default is 'slen'.
        frames : None, (start_s, stop_s) or sequence of int, optional
            Frames to render. Default all tracked frames.
        background : {'image', 'zbands', 'none'}, optional
            Background layer. Default is 'image'.
        cmap, clim, linewidth : optional
            As in :meth:`Plots.plot_track_vectors`.
        highlight_groups : dict, optional
            ``{'myofibril': 8, 'mband': 39}``: members of these groups are outlined.
        title : str, optional
            Panel title; default the quantity label.
        fps, slow_motion, dpi, panel_width, progress : optional
            As in :meth:`animate_image`.

        Returns
        -------
        str
            The output path.
        """
        frames = Animations.resolve_frames(sarc_obj, frames)
        kin = sarc_obj.get_track_kinematics() if color_by in ('delta_slen', 'vel') else None
        h, w = sarc_obj.metadata.size[-2:]
        ph = panel_width * h / w
        fig = plt.figure(figsize=(panel_width + 1.0, ph + 0.6), dpi=dpi)
        ax = fig.add_axes([0.3 / (panel_width + 1.0), 0.1 / (ph + 0.6), panel_width / (panel_width + 1.0), ph / (ph + 0.6)])
        renderers = []
        if background != 'none':
            style = 'bands' if (background == 'zbands' and color_by in ('delta_slen', 'vel')) else 'mask'
            bg = _ImageLayer(sarc_obj, frames, layer=background, style=style)
            bg.preload()
            bg.setup(ax, int(frames[0]))
            renderers.append(bg)
        highlight = []
        for kind, g in (highlight_groups or {}).items():
            gid, _ = Animations._group_members(sarc_obj, kind)
            highlight.append(np.flatnonzero(gid == g))
        vec = _TrackVectorLayer(sarc_obj, color_by=color_by, cmap=cmap, clim=clim, linewidth=linewidth,
                                highlight=highlight, kinematics=kin)
        vec.setup(ax, int(frames[0]))
        renderers.append(vec)
        ax.set_title(title or Plots.TRACK_COLOR_DEFAULTS[color_by][2], fontsize=PlotUtils.fontsize, fontweight='bold')
        return Animations._run(fig, renderers, frames, out, Animations.playback_fps(sarc_obj, fps, slow_motion), dpi, progress)

    @staticmethod
    def animate_results(sarc_obj: SarcAsM, out: str, frames: FramesSpec = (0.0, 8.0),
                        image_panels: Sequence[str] = ('image', 'zbands', 'vectors:slen', 'vectors:delta_slen'),
                        timeseries: Sequence[str] = ('pool', 'mband:auto', 'myofibril:auto'),
                        quantity: str = 'slen', y_lim: Optional[Tuple[float, float]] = None,
                        percentiles: Tuple[float, ...] = (5, 25, 75, 95), n_faint: int = 60,
                        panel_labels: bool = True, vector_background: str = 'image', min_coverage: float = 0.8,
                        fps: Optional[float] = None, slow_motion: float = 2.4,
                        dpi: int = 160, figwidth: float = 9.0, timeseries_height: float = 1.3,
                        progress: bool = False, snapshot: bool = False) -> str:
        """
        Composite results movie: image panels on top (raw movie, Z-band masks, live sarcomere-vector
        colormaps) and grouped time series below with a cursor at the current time.

        Parameters
        ----------
        sarc_obj : SarcAsM
            Object with tracks and a pooled motion analysis (``analyze_track_motion(by='pool')``);
            the M-band / myofibril groupings for the requested time series.
        out : str
            Output path (``.mp4`` or ``.gif``).
        frames : None, (start_s, stop_s) or sequence of int, optional
            Window to render. Default the first 8 s.
        image_panels : sequence of str, optional
            ``'image'``, ``'zbands'`` or ``'vectors:<quantity>'`` (quantity as in
            :meth:`Plots.plot_track_vectors`), laid out in two columns. Default shows all four.
        timeseries : sequence of str, optional
            ``'pool'``, ``'mband:auto'`` / ``'mband:<id>'``, ``'myofibril:auto'`` / ``'myofibril:extent'`` /
            ``'myofibril:<id>'``;
            ``auto`` picks the group with the most members observed in every frame of the window
            (see :meth:`auto_group`). The groups shown are outlined on the vector panels.
        quantity : {'slen', 'delta_slen'}, optional
            Time-series quantity. Default is 'slen'.
        y_lim : tuple of float, optional
            Common y-limits of the time-series panels; default from the data.
        percentiles : tuple of 4 floats, optional
            Outer and inner band of the pool panel. Default is (5, 25, 75, 95).
        n_faint : int, optional
            Number of random member traces drawn faintly in the pool panel. Default is 60.
        panel_labels : bool, optional
            Bold lower-case panel letters. Default is True.
        vector_background : {'image', 'zbands'}, optional
            Background of the vector panels: the raw movie (default) or the Z-band mask drawn as dark
            bands on grey.
        min_coverage : float, optional
            Members of the time-series groups must be observed in this fraction of the window frames
            (gaps are drawn as breaks). Default is 0.8.
        fps : float, optional
            Playback rate; default real time.
        slow_motion : float, optional
            Slow-down factor. Default is 2.4.
        dpi : int, optional
            Resolution. Default is 160.
        figwidth : float, optional
            Figure width in inches. Default is 9.
        timeseries_height : float, optional
            Height of the time-series row in inches. Default is 1.3.
        progress : bool, optional
            Print progress every 100 frames. Default is False.
        snapshot : bool, optional
            Also save the first frame as ``<out>_frame0.png``. Default is False.

        Returns
        -------
        str
            The output path.
        """
        frames = Animations.resolve_frames(sarc_obj, frames)
        ft = float(sarc_obj.metadata.frametime)
        kin = sarc_obj.get_track_kinematics()
        groups = [Animations._parse_group(sarc_obj, spec, frames, min_coverage) for spec in timeseries]
        highlight = [members for kind, _, members in groups if kind != 'pool']

        h, w = sarc_obj.metadata.size[-2:]
        n_img = len(image_panels)
        ncol = 2 if n_img > 1 else 1
        nrow = int(np.ceil(n_img / ncol))
        pw = (figwidth - 0.35 - 0.45 * ncol) / ncol
        ph = pw * h / w
        top_h = 0.35 + nrow * (ph + 0.42)
        ts_top = top_h + 0.5
        fh = ts_top + timeseries_height + 0.45
        fig = plt.figure(figsize=(figwidth, fh), dpi=dpi)
        def ax_(x, y, ww, hh):
            return fig.add_axes([x / figwidth, 1 - (y + hh) / fh, ww / figwidth, hh / fh])
        letters = iter('abcdefghijklmnopqrstuvwxyz')
        renderers = []
        titles = {'image': 'Confocal movie', 'zbands': 'Z-bands (3D U-Net)', 'vectors:slen': 'Sarcomere length SL',
                  'vectors:delta_slen': 'Length change ΔSL', 'vectors:vel': 'Velocity V',
                  'vectors:coverage': 'Track coverage', 'vectors:group': 'Groups'}
        for i, spec in enumerate(image_panels):
            col, row = i % ncol, i // ncol
            ax = ax_(0.35 + col * (pw + 0.45), 0.35 + row * (ph + 0.42), pw, ph)
            if spec in ('image', 'zbands'):
                r = _ImageLayer(sarc_obj, frames, layer=spec)
                r.preload()
                r.setup(ax, int(frames[0]))
                renderers.append(r)
            else:
                q = spec.split(':', 1)[1]
                bg = _ImageLayer(sarc_obj, frames, layer=vector_background,
                                 style='bands' if vector_background == 'zbands' else 'mask')
                bg.preload()
                bg.setup(ax, int(frames[0]))
                renderers.append(bg)
                vec = _TrackVectorLayer(sarc_obj, color_by=q, highlight=highlight, kinematics=kin)
                vec.setup(ax, int(frames[0]))
                renderers.append(vec)
            ax.set_title(titles.get(spec, spec), fontsize=PlotUtils.fontsize, fontweight='bold', pad=3)
            if panel_labels:
                bb = ax.get_position()
                fig.text(bb.x0 - 0.025, bb.y1 + 0.005, next(letters), fontsize=PlotUtils.fontsize + 3,
                         fontweight='bold', va='bottom')
        time_label = fig.text(0.5, 1 - (top_h + 0.1) / fh, '', ha='center', fontsize=PlotUtils.fontsize,
                              fontweight='bold')

        n_ts = len(groups)
        gap = 0.5
        tw = (figwidth - 0.6 - 0.25 - (n_ts - 1) * gap) / n_ts
        ts_axes = []
        for i, (kind, g, members) in enumerate(groups):
            ax = ax_(0.6 + i * (tw + gap), ts_top, tw, timeseries_height)
            if kind == 'pool':
                title = f'whole cell (n = {members.size:,})'
            elif kind == 'mband':
                title = f'one M-band group ({members.size} positions)'
            else:
                title = f'one {kind} (n = {members.size})'
            Animations._draw_group_overlay(ax, sarc_obj, kind, members, frames, quantity, kin, percentiles,
                                           n_faint, y_lim=y_lim, title=title)
            PlotUtils.polish_xticks(ax, pad=2)
            ts_axes.append(ax)
            if panel_labels:
                bb = ax.get_position()
                fig.text(bb.x0 - 0.045, bb.y1 + 0.005, next(letters), fontsize=PlotUtils.fontsize + 3,
                         fontweight='bold', va='bottom')
        if y_lim is None and ts_axes:
            lo = min(a.get_ylim()[0] for a in ts_axes)
            hi = max(a.get_ylim()[1] for a in ts_axes)
            for a in ts_axes:
                a.set_ylim(lo, hi)
        for a in ts_axes:
            major, minor = PlotUtils.nice_tick_steps(float(np.ptp(a.get_ylim())), target=4)
            PlotUtils.polish_yticks(a, major, minor, pad=2)
        cursor = _TimeCursor(frames * ft, ts_axes, time_label)
        cursor.setup(None, 0)
        # the cursor indexes positions within ``frames``
        pos_of = {int(f): i for i, f in enumerate(frames)}

        class _CursorAdapter:
            def update(self, f):
                cursor.update(pos_of[int(f)])
        renderers.append(_CursorAdapter())
        return Animations._run(fig, renderers, frames, out, Animations.playback_fps(sarc_obj, fps, slow_motion), dpi,
                               progress, snapshot)
