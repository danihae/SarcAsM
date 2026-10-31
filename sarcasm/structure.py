# -*- coding: utf-8 -*-
# Copyright (c) 2025 University Medical Center Göttingen, Germany.
# All rights reserved.
#
# Patent Pending: DE 10 2024 112 939.5
# SPDX-License-Identifier: LicenseRef-Proprietary-See-LICENSE
#
# This software is licensed under a custom license. See the LICENSE file
# in the root directory for full details.
#
# **Commercial use is prohibited without a separate license.**
# Contact MBM ScienceBridge GmbH (https://sciencebridge.de/en/) for licensing.


"""``sarc.structure``: sarcomere morphology of a recording (Z-bands, sarcomere vectors, myofibrils, domains)."""

import logging
from typing import List, Optional, Sequence, Tuple, Union

import warnings
import numpy as np
from bio_image_unet.progress import ProgressNotifier
from scipy import stats, sparse

from sarcasm.utils import Utils


logger = logging.getLogger(__name__)

# Import structure modules
from sarcasm.analysis import (
    z_band_analysis,
    sarcomere_vectors,
    myofibril_analysis,
    domain_clustering,
)

from sarcasm._internal.namespace import _Namespace


class StructureAnalysis(_Namespace):
    """Sarcomere morphology of one recording, reached as ``sarc.structure``.

    Each step reads the detection masks of :meth:`SarcAsM.detect_sarcomeres
    <sarcasm.core.SarcAsM.detect_sarcomeres>` and writes its results to ``sarc.data``
    under ``structure.*`` (and its parameters under ``params.structure.*``).
    """

    def _plan_frames(self, frames, step: str) -> List[int]:
        """The frames an analysis step that reads the detection masks analyzes: ``'all'`` = every detected
        frame, otherwise the canonical ``frames``, which must all have been detected. Masks and raw images are
        read, and results filed, by frame number.

        Raises
        ------
        ValueError
            For frames that were not detected, naming the detected frames.
        """
        n_stack = self.metadata.n_stack or 1
        requested = self._canonical_frames(frames)
        detected = self._detected_frames()
        detected = sorted(set(detected)) if detected is not None else list(range(n_stack))
        if requested == 'all':
            if detected != list(range(n_stack)):
                logger.info(f'{step}: analyzing the {len(detected)} detected frame(s) {detected} of {n_stack}.')
            return detected
        missing = [f for f in requested if f not in detected]
        if missing:
            raise ValueError(f"{step}: frame(s) {missing} were not detected (detect_sarcomeres covered frames "
                             f"{detected}). Run detect_sarcomeres(frames=...) on them first.")
        return requested

    def _read_raw_frames(self, frames: List[int]) -> np.ndarray:
        """Raw images of the movie frames ``frames`` as a ``(len(frames), H, W)`` stack (single images too)."""
        if (self.metadata.n_stack or 1) <= 1:
            return np.asarray(self.read_imgs()).reshape(1, *self.metadata.size)
        return np.asarray(self.read_imgs(frames=list(frames)))

    def analyze_cell_mask(self, frames: Union[str, int, Sequence[int]] = 'all', threshold: float = 0.1) -> None:
        """
        Analyze the area occupied by cells and compute average cell intensity and
        cell area ratio.

        Runs automatically at the end of :meth:`detect_sarcomeres` for the detected
        frames; call it directly only to re-evaluate with another ``threshold``.

        Parameters
        ----------
        frames : {'all', int, sequence of int} or None, optional
            Frames to analyze: ``'all'`` (or None) for every detected frame, a frame
            number, or any sequence of frame numbers; they must have been detected.
            Results are stored under the frame numbers. Default is 'all'.
        threshold : float, optional
            Threshold for binarizing the cell mask; pixels above it are cell.
            Default is 0.1.
        """
        if not self.mask_exists('cell_mask'):
            raise FileNotFoundError("Cell mask not found. Please run detect_sarcomeres first.")
        list_frames = self._plan_frames(frames, 'analyze_cell_mask')
        cell_mask = self.read_mask('cell_mask', frames=list_frames)
        images = self._read_raw_frames(list_frames)

        # per-frame arrays indexed by frame number (NaN for frames not analyzed), like every other
        # per-frame structure result, so that analyze_z_bands / analyze_vectors can index
        # them by frame when only a subset of frames is analyzed
        n_stack = self.metadata.n_stack
        cell_area, cell_area_ratio = np.full(n_stack, fill_value=np.nan), np.full(n_stack, fill_value=np.nan)
        cell_mask_intensity = np.full(n_stack, fill_value=np.nan)

        for i, img_i, cell_mask_i in zip(list_frames, images, cell_mask):
            # binarize mask
            mask_i = cell_mask_i > threshold

            # average cell intensity
            cell_mask_intensity[i] = np.mean(img_i[mask_i])

            # total cell area and ratio to total image area
            if self.metadata.pixelsize is not None:
                cell_area[i] = np.sum(mask_i) * self.metadata.pixelsize ** 2
                cell_area_ratio[i] = cell_area[i] / (img_i.shape[0] * img_i.shape[1] * self.metadata.pixelsize ** 2)

        _dict = {'structure.cell.mask_area': cell_area, 'structure.cell.mask_area_ratio': cell_area_ratio,
                 'structure.cell.mask_intensity': cell_mask_intensity,
                 'params.structure.analyze_cell_mask.frames': list_frames,
                 'params.structure.analyze_cell_mask.threshold': threshold}
        self.data.update(_dict)
        if self.auto_save:
            self.commit()

    def _recording_background(self, n_frames: int = 5) -> float:
        """Background intensity of the recording: the median over up to ``n_frames`` evenly spaced detected frames of
        the median intensity outside their cell masks (camera offset plus medium, constant within a recording). It
        does not depend on which frames an analysis call covers. NaN without free background in any of them."""
        frames = sorted(set(self._detected_frames() or range(self.metadata.n_stack or 1)))
        picks = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).round().astype(int)
        frames = sorted({frames[i] for i in picks})
        masks = self.read_mask('cell_mask', frames=frames) > self.data.get('params.structure.analyze_cell_mask.threshold', 0.1)
        images = self._read_raw_frames(frames)
        values = np.array([z_band_analysis.estimate_background(image, mask) for image, mask in zip(images, masks)])
        return float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan

    def analyze_z_bands(self, frames: Union[str, int, Sequence[int]] = 'all', threshold: float = 0.5,
                        min_length: float = 0.2, median_filter_radius: float = 0.2, theta_phi_min: float = 0.4, 
                        a_min: float = 0.3, d_max: float = 3.0, d_min: float = 0.0, max_axial_offset: float = 0.9,
                        progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Segment and analyze sarcomere z-bands.

        Besides the geometry and lateral connections of the z-bands, each z-band gets a ratiometric
        contrast (``structure.zbands.contrast``): its background-subtracted mean intensity divided by
        that of the cell pixels within 1 µm around it. The background is the median intensity outside
        the cell mask of the frame (``structure.zbands.background``). A frame (almost) fully covered by
        cells has no free background and takes the background of the recording, the median over up to
        five evenly spaced detected frames, as the camera offset is constant within a recording; only
        when none of these has free background is the 1st percentile of the frame used.

        Parameters
        ----------
        frames : {'all', int, sequence of int} or None, optional
            Frames to analyze: ``'all'`` (or None) for every detected frame, a frame
            number, or any sequence of frame numbers; they must have been detected.
            Results are stored under the frame numbers. Default is 'all'.
        threshold : float, optional
            Threshold for binarizing z-bands prior to labeling (0-1).
            Default is 0.5.
        min_length : float, optional
            Minimal z-band length in µm; shorter z-bands are removed.
            Default is 0.2.
        median_filter_radius : float, optional
            Radius of the kernel smoothing the orientation field, in µm.
            Default is 0.2.
        theta_phi_min : float, optional
            Minimal cosine of the angle between the pointed z-band vector and the
            vector connecting z-band ends; smaller values are not recognized as
            connections.
            Default is 0.4.
        a_min : float, optional
            Minimal linking score of two z-band ends (product of the cosines between
            the end directions and the connecting line, 0-1) to create a lateral
            connection. Default is 0.3.
        d_max : float, optional
            Maximal distance between z-band ends in µm; pairs farther apart are
            not connected. Default is 3.0.
        d_min : float, optional
            Minimal distance between z-band ends in µm; pairs closer than this are
            not connected. Default is 0.0.
        max_axial_offset : float or None, optional
            Maximal offset in µm of a lateral connection along the sarcomere axis
            (perpendicular to the z-bands); a connection offset by about half a
            sarcomere or more reaches the next z-band of the neighbouring myofibril.
            None disables the limit. Default is 0.9.
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in the GUI. Default is
            ProgressNotifier.progress_notifier_tqdm().
        """
        if not self.mask_exists('zbands'):
            raise FileNotFoundError("Z-band mask not found. Please run detect_sarcomeres first.")
        list_frames = self._plan_frames(frames, 'analyze_z_bands')
        zbands = self.read_mask('zbands', frames=list_frames)
        # orientation is stored (frames, 2, H, W) — read by frame like zbands.
        orientation_field = self.read_mask('orientation', frames=list_frames)
        images = self._read_raw_frames(list_frames)
        # the cell mask sets the background of the Z-band contrast; binarized like analyze_cell_mask
        cell_mask = None
        if self.mask_exists('cell_mask'):
            cell_mask = (self.read_mask('cell_mask', frames=list_frames)
                         > self.data.get('params.structure.analyze_cell_mask.threshold', 0.1))
        # One mask page per detected frame — see the same guard in
        # analyze_vectors.
        n_imgs = min(len(zbands), len(images), len(orientation_field),
                     len(cell_mask) if cell_mask is not None else len(zbands))
        if n_imgs < len(list_frames):
            logger.warning(
                f'detect_sarcomeres covered {n_imgs} frame(s), but {len(list_frames)} were '
                f'requested — analyzing the first {n_imgs}. Run detect_sarcomeres on the '
                'frames you want to analyze.')
            list_frames = list_frames[:n_imgs]

        # background of the Z-band contrast: the median outside the cell mask of each frame; frames (almost)
        # fully covered by cells take the background of the recording (see _recording_background), and without
        # any free background the module falls back to the 1st percentile of the frame
        backgrounds = np.full(n_imgs, np.nan)
        if cell_mask is not None:
            backgrounds = np.array([z_band_analysis.estimate_background(images[k], cell_mask[k])
                                    for k in range(n_imgs)])
            if not np.isfinite(backgrounds).all():
                backgrounds[~np.isfinite(backgrounds)] = self._recording_background()

        # create empty lists
        def none_lists():
            return [None] * self.metadata.n_stack
        z_length, z_intensity, z_contrast, z_orientation = (none_lists() for _ in range(4))
        z_lat_neighbors, z_lat_dist, z_lat_length_groups = (none_lists() for _ in range(3))
        z_labels, z_ends, z_lat_links, z_lat_groups = (none_lists() for _ in range(4))

        # create empty arrays
        def nan_arrays():
            return np.full(self.metadata.n_stack, np.nan)
        z_length_mean, z_length_std, z_length_max, z_oop = (nan_arrays() for _ in range(4))
        n_zbands, z_intensity_mean, z_intensity_std = (nan_arrays() for _ in range(3))
        z_contrast_median, z_surround_median, z_background = nan_arrays(), nan_arrays(), nan_arrays()
        z_mask_area, z_mask_intensity, z_mask_area_ratio = (nan_arrays() for _ in range(3))
        z_lat_neighbors_mean, z_lat_neighbors_std = (nan_arrays() for _ in range(2))
        z_lat_dist_mean, z_lat_dist_std = (nan_arrays() for _ in range(2))
        z_lat_length_groups_mean, z_lat_length_groups_std = (nan_arrays(), nan_arrays())

        # iterate images
        logger.info('Starting Z-band analysis...')
        for i, (frame_i, zbands_i, image_i, orientation_field_i) in enumerate(
                progress_notifier.iterator(zip(list_frames, zbands, images, orientation_field), total=n_imgs)):

            # Delegate to z_band_analysis module
            labels_i, labels_skel_i = z_band_analysis.segment_z_bands(zbands_i, threshold=threshold)

            # analyze z-band features
            res = z_band_analysis.analyze_z_bands(
                zbands_i, labels_i, labels_skel_i, image_i, orientation_field_i,
                pixelsize=self.metadata.pixelsize, threshold=threshold,
                min_length=min_length, median_filter_radius=median_filter_radius,
                a_min=a_min, theta_phi_min=theta_phi_min,
                d_max=d_max, d_min=d_min, max_axial_offset=max_axial_offset, background=backgrounds[i],
                cell_mask=cell_mask[i] if cell_mask is not None else None
            )
            z_length_i, z_intensity_i, z_contrast_i = res['length'], res['intensity'], res['contrast']
            z_lat_neighbors_i, z_lat_dist_i = res['lat_neighbors'], res['lat_dist']
            z_lat_length_groups_i = res['lat_length_groups']

            # fill lists and arrays
            z_length[frame_i] = z_length_i
            z_intensity[frame_i] = z_intensity_i
            z_contrast[frame_i] = z_contrast_i
            z_lat_neighbors[frame_i] = z_lat_neighbors_i
            z_orientation[frame_i] = res['orientation']
            z_lat_dist[frame_i] = z_lat_dist_i
            z_lat_length_groups[frame_i] = z_lat_length_groups_i
            z_mask_area[frame_i], z_mask_intensity[frame_i], z_oop[frame_i] = (res['mask_area'],
                                                                               res['mask_intensity'], res['oop'])
            z_background[frame_i] = res['background']
            if 'structure.cell.mask_area' in self.data:
                z_mask_area_ratio[frame_i] = res['mask_area'] / self.data['structure.cell.mask_area'][frame_i]
            else:
                z_mask_area_ratio[frame_i] = res['mask_area'] / (self.metadata.size[0] * self.metadata.size[1])

            z_labels[frame_i] = sparse.coo_matrix(res['labels'])
            z_lat_links[frame_i] = res['links']
            z_ends[frame_i] = res['ends']
            z_lat_groups[frame_i] = res['lat_groups']

            # calculate mean and std of z-band features
            if len(z_length_i) > 0:
                z_length_mean[frame_i], z_length_std[frame_i], z_length_max[frame_i] = (
                    np.mean(z_length_i), np.std(z_length_i), np.max(z_length_i))
            n_zbands[frame_i] = len(z_length_i)
            z_intensity_mean[frame_i], z_intensity_std[frame_i] = np.mean(z_intensity_i), np.std(z_intensity_i)
            if np.isfinite(z_contrast_i).any():
                z_contrast_median[frame_i] = np.nanmedian(z_contrast_i)
            if np.isfinite(res['surround']).any():
                z_surround_median[frame_i] = np.nanmedian(res['surround'])
            z_lat_neighbors_mean[frame_i], z_lat_neighbors_std[frame_i] = np.mean(z_lat_neighbors_i), np.std(
                z_lat_neighbors_i)
            z_lat_dist_mean[frame_i], z_lat_dist_std[frame_i] = np.nanmean(z_lat_dist_i), np.nanstd(z_lat_dist_i)
            z_lat_length_groups_mean[frame_i], z_lat_length_groups_std[frame_i] = np.nanmean(
                z_lat_length_groups_i), np.nanstd(
                z_lat_length_groups_i)

        # create and save dictionary for cell structure
        z_band_data = {'structure.zbands.n': n_zbands, 'structure.zbands.length': z_length, 'structure.zbands.length_mean': z_length_mean, 'structure.zbands.length_std': z_length_std,
                       'structure.zbands.length_max': z_length_max, 'structure.zbands.intensity': z_intensity, 'structure.zbands.intensity_mean': z_intensity_mean,
                       'structure.zbands.intensity_std': z_intensity_std, 'structure.zbands.contrast': z_contrast,
                       'structure.zbands.contrast_median': z_contrast_median,
                       'structure.zbands.surround_intensity_median': z_surround_median,
                       'structure.zbands.background': z_background,
                       'structure.zbands.orientation': z_orientation, 'structure.zbands.oop': z_oop,
                       'structure.zbands.mask_intensity': z_mask_intensity, 'structure.zbands.labels': z_labels,
                       'structure.zbands.mask_area': z_mask_area, 'structure.zbands.mask_area_ratio': z_mask_area_ratio, 'structure.zbands.lat_neighbors': z_lat_neighbors,
                       'structure.zbands.lat_neighbors_mean': z_lat_neighbors_mean, 'structure.zbands.lat_neighbors_std': z_lat_neighbors_std,
                       'structure.zbands.lat_dist': z_lat_dist, 'structure.zbands.ends': z_ends,
                       'structure.zbands.lat_dist_mean': z_lat_dist_mean, 'structure.zbands.lat_dist_std': z_lat_dist_std, 'structure.zbands.lat_links': z_lat_links,
                       'structure.zbands.lat_groups': z_lat_groups,
                       'structure.zbands.lat_length_groups': z_lat_length_groups,
                       'structure.zbands.lat_length_groups_mean': z_lat_length_groups_mean,
                       'structure.zbands.lat_length_groups_std': z_lat_length_groups_std,
                       'params.structure.analyze_z_bands.frames': list_frames, 'params.structure.analyze_z_bands.threshold': threshold,
                       'params.structure.analyze_z_bands.min_length': min_length, 'params.structure.analyze_z_bands.median_filter_radius': median_filter_radius,
                       'params.structure.analyze_z_bands.theta_phi_min': theta_phi_min, 'params.structure.analyze_z_bands.d_max': d_max,
                       'params.structure.analyze_z_bands.d_min': d_min, 'params.structure.analyze_z_bands.max_axial_offset': max_axial_offset}
        self.data.update(z_band_data)
        if self.auto_save:
            self.commit()

    def analyze_vectors(self, frames: Union[str, int, Sequence[int]] = 'all', threshold_mbands: float = 0.25,
                                  median_filter_radius: float = 0.25, linewidth: float = 0.3, interp_factor: int = 4,
                                  slen_lims: Tuple[float, float] = (1, 3), threshold_sarcomere_mask=0.1,
                                  interpolation_method: str = 'akima',
                                  smooth_orientation_sigma: float = 0.0,
                                  peak_prominence: float = 0.4,
                                  neighbor_radius: float = 1.0,
                                  max_neighbor_deviation: Optional[float] = 30.0,
                                  registry_lateral_band: Tuple[float, float] = (1.5, 2.5),
                                  registry_axial_range: int = 3,
                                  registry_max_angle: float = 20.0,
                                  use_fast_movie_zbands: bool = True,
                                  progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Extract sarcomere orientation and length vectors, and the Z-band registry of every vector.

        Parameters
        ----------
        frames : {'all', int, sequence of int} or None, optional
            Frames to analyze: ``'all'`` (or None) for every detected frame, a frame
            number, or any sequence of frame numbers; they must have been detected.
            Results are stored under the frame numbers. Default is 'all'.
        threshold_mbands : float, optional
            Threshold to binarize sarcomere M-bands. Lower values may yield more
            false-positive sarcomere vectors. Default is 0.25.
        median_filter_radius : float, optional
            Radius of the kernel smoothing the orientation field before assessing
            orientation at M-points, in µm. Default is 0.25.
        linewidth : float, optional
            Width of the profile lines used to measure sarcomere length, in µm.
            Default is 0.3. The averaging runs *along* the M-band, i.e. laterally
            across the myofibril, so wider lines pool neighbouring sarcomeres.

            Widening it does **not** meaningfully reduce frame-to-frame
            variability. Measured at fixed geometry, 0.1 → 1.5 µm moves the noise
            only 32.4 → 27.1 nm (−16 %), against the −72 % that averaging 12.5×
            more independent samples would give. The reason is that the residual
            noise is a coherent wobble of the whole Z-band: the temporal residuals
            of two *disjoint* lateral half-windows correlate at 0.85 / 0.68 / 0.42
            at 0.15 / 0.34 / 0.61 µm separation. Lateral averaging therefore
            re-samples the same fluctuation instead of cancelling it. (Per-pixel
            segmentation flicker does decorrelate over ~0.5 µm, but the
            centre-of-mass along the profile has already averaged that away.)

            What widening does buy is a slightly higher detection rate (~+1.5 %
            from 0.3 to 1.5 µm), and what it costs is real: two disjoint halves of
            the window agree to ~30 nm at 0.3 µm but disagree by ~80 nm at 1.5 µm,
            i.e. a wide line pools sarcomeres that genuinely differ. The mean
            length shifts by ~1 % over that range with a *sample-dependent sign*
            (+18 nm on one recording, −20 nm on two others), so absolute lengths
            are not comparable between analyses run at different ``linewidth``.
            Runtime scales ~4.6× from 0.2 to 1.5 µm.

            In short, treat it as a detection-rate knob with an accuracy cost, not
            as a noise filter. To actually reduce frame-to-frame length noise on
            high-speed recordings, run :meth:`~sarcasm.motion.MotionAnalysis.detect_z_bands_fast_movie` — the 3D
            U-Net cuts it ~4× (32 → 8 nm) by integrating over a temporal window
            during prediction, without smearing moving Z-bands after the fact.
        interp_factor : int, optional
            Akima/linear upsampling factor applied to each profile before peak
            detection. Default is 4. Higher values refine sub-pixel peak
            localisation, which sets the resolution of the length estimate.
        slen_lims : tuple of float, optional
            Sarcomere length limits in µm. Default is (1, 3).
        threshold_sarcomere_mask : float, optional
            Threshold to binarize the sarcomere mask before measuring
            ``sarcomere_area``. Default is 0.1, matching the default of
            :meth:`analyze_cell_mask`, so that ``sarcomere_area_ratio`` divides
            two comparably-thresholded areas.
        interpolation_method : str, optional
            Interpolation method for profile analysis: 'linear' (fast) or 'akima'
            (smooth). Default is 'akima'.
        smooth_orientation_sigma : float, optional
            Temporal Gaussian sigma (in frames) for smoothing the orientation
            field along the time axis before per-frame vector extraction,
            reducing frame-to-frame jitter (axially correct via the double-angle
            trick). 0 disables smoothing; ``sigma ≈ 1`` is a ~5-frame effective
            span. Default is 0.0.

            Only meaningful for high-speed recordings, where consecutive frames
            image the same structures. Long-term time-lapses have no frame-to-frame
            correspondence, so smoothing across their frames is meaningless — leave
            this at 0 there and use ``linewidth`` instead.

            The spatial ``median_filter_radius`` already removes most per-pixel
            orientation jitter; this suppresses the residual frame-to-frame
            component. Because orientation enters the length estimate only through
            the cosine of a small angular error, smoothing it beyond a modest sigma
            does not improve length. To reduce frame-to-frame length variability,
            run :meth:`~sarcasm.motion.MotionAnalysis.detect_z_bands_fast_movie` and let ``use_fast_movie_zbands``
            pick up its output.
        peak_prominence : float, optional
            ``scipy.signal.find_peaks`` prominence threshold for Z-band peak
            detection inside each profile, applied after each profile is min-max
            normalised. Lower values accept weaker, noisier peaks and recover
            detections where Z-band contrast is reduced, such as at peak
            contraction or during fast motion. Default is 0.4.

            It decides *which* vectors are admitted, not how precisely a peak is
            localised: lengths agree exactly on every vector two settings both
            accept. Below ~0.4 a tail of poorly conditioned lengths appears (they
            inflate ``sarcomere_length_std`` and leak outliers into tracking);
            above it only the detection count falls, faster on dim recordings.
            Lower it if your recordings are dim.
        neighbor_radius : float, optional
            Radius, in µm, of the neighbourhood against which each vector's
            orientation is checked (see ``max_neighbor_deviation``). Default is
            1.0, about the extent of one M-band, so a vector is compared with
            the rest of its own M-band and the ends of the adjacent ones.
        max_neighbor_deviation : float or None, optional
            Vectors whose orientation deviates by more than this angle, in
            degrees, from the mean orientation of the vectors within
            ``neighbor_radius`` are removed. Default is 30.0; ``None`` keeps
            every vector. The number removed per frame is stored in
            ``structure.sarcomere.n_vectors_removed``.
        registry_lateral_band : tuple of float, optional
            Z-band registry: distance across the sarcomere axis, in µm, within which a
            neighbouring sarcomere counts as one myofibril over. Default is (1.5, 2.5).
        registry_axial_range : int, optional
            Z-band registry: farthest neighbour along the myofibril, in sarcomeres.
            Default is 3.
        registry_max_angle : float, optional
            Z-band registry: largest axial angle, in degrees, between a neighbour along
            the myofibril and the vector for it to count. Default is 20.0.

            The registry of a sarcomere vector is the mean cosine of the phase of the
            neighbouring Z-discs on its own Z-disc lattice (the vector's two Z-discs
            repeated every sarcomere length): 1 = the neighbours' Z-discs are level with
            its own, 0 = random, −1 = they sit at its M-band. The lateral registry
            uses the neighbours one myofibril over, the axial registry the neighbours
            along the myofibril; see :func:`sarcasm.analysis.sarcomere_vectors.zband_registry`.
            Stored per vector (``structure.sarcomere.registry_lateral`` / ``registry_axial``,
            NaN without neighbours) and per frame as the means
            (``registry_lateral_mean``, ``registry_axial_mean`` and their average
            ``registry_mean``).

            This removes false-positive vectors: M-band fragments pointing
            across their neighbours, typically at crossings of differently
            oriented myofibrils, and stray detections. On recordings from five
            sources at 0.06–0.16 µm/px the default removes 0–2 % of the vectors
            and none on long, ordered M-band combs. The rule reads only vector
            positions and orientations; a vector without any neighbour inside
            the radius (a thin or short M-band) is always kept, so nothing
            hinges on how many points were detected on an M-band.
        use_fast_movie_zbands : bool, optional
            If True and a ``zbands_fast_movie`` mask exists (produced by
            :meth:`~sarcasm.motion.MotionAnalysis.detect_z_bands_fast_movie`), use that 3D U-Net output instead
            of the per-frame ``zbands`` mask. Because it is predicted from a
            temporal window, it varies less between consecutive frames, which
            carries through to the measured lengths. Falls back to the 2D mask when
            unavailable; the choice is stored in
            ``params.structure.analyze_vectors.zbands_source``. Default is True.
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in the GUI. Default is
            ProgressNotifier.progress_notifier_tqdm().
        """
        if not self.mask_exists('zbands'):
            raise FileNotFoundError("Z-band mask not found. Please run detect_sarcomeres first.")

        # Decide which Z-band stack to use (fast-movie 3D U-Net vs per-frame 2D).
        if use_fast_movie_zbands and self.mask_exists('zbands_fast_movie'):
            zbands_name = 'zbands_fast_movie'
            zbands_source = 'fast_movie_3d'
        else:
            zbands_name = 'zbands'
            if use_fast_movie_zbands:
                zbands_source = 'per_frame_2d'
                # the normal case (the 3D fast-movie Z-bands are optional), so not shown by default
                logger.debug(
                    'analyze_vectors: using the per-frame 2D U-Net Z-bands; the 3D '
                    'fast-movie Z-bands (motion.detect_z_bands_fast_movie) were not computed.'
                )
            else:
                zbands_source = 'per_frame_2d_forced'
                logger.info(
                    'analyze_vectors: using the per-frame 2D U-Net Z-bands '
                    '(use_fast_movie_zbands=False).'
                )

        # All inputs are read by frame number: the 2D masks map it to their pages, the fast-movie Z-bands (one
        # page per movie frame) are read at the frame itself, so every input of a frame comes from that frame.
        list_frames = self._plan_frames(frames, 'analyze_vectors')
        z_bands = self.read_mask(zbands_name, frames=list_frames)
        mbands = self.read_mask('mbands', frames=list_frames)
        orientation_field = self.read_mask('orientation', frames=list_frames)
        sarcomere_mask = self.read_mask('sarcomere_mask', frames=list_frames)

        # a partial detection has fewer mask pages than frames requested; refuse
        # rather than let zip() silently truncate the analysis
        stack_lengths = {'zbands': len(z_bands), 'mbands': len(mbands),
                         'orientation': len(orientation_field),
                         'sarcomere_mask': len(sarcomere_mask)}
        n_available = min(stack_lengths.values())
        if n_available < len(list_frames):
            shortest = ', '.join(f'{name} ({n})' for name, n in stack_lengths.items()
                                 if n == n_available)
            logger.warning(
                f'Only {n_available} of the {len(list_frames)} requested frames have masks for '
                f'every input (shortest: {shortest}) — analyzing the first {n_available}. '
                'M-bands, sarcomere orientation and the sarcomere mask are needed per frame '
                'and only detect_sarcomeres produces them (motion.detect_z_bands_fast_movie replaces '
                'the Z-band mask alone). For motion analysis, run detect_sarcomeres on every '
                'frame you want to track.')
            list_frames = list_frames[:n_available]

        # Optional temporal smoothing of the orientation field.
        if smooth_orientation_sigma > 0 and orientation_field.shape[0] > 1:
            logger.info(
                f'Temporally smoothing orientation field with sigma={smooth_orientation_sigma:.3f} frames...'
            )
            orientation_field = sarcomere_vectors.smooth_orientation_field_temporal(
                orientation_field, sigma=smooth_orientation_sigma,
            )

        n_frames = len(list_frames)
        pixelsize = self.metadata.pixelsize

        # Check pixelsize is not None
        if pixelsize is None:
            raise ValueError("Pixel size is not available. Please provide pixelsize during initialization.")

        # create empty arrays
        def none_lists():
            return [None] * self.metadata.n_stack
        def nan_arrays():
            return np.full(self.metadata.n_stack, np.nan)
        (pos_vectors, pos_vectors_px, sarcomere_length_vectors,
         sarcomere_orientation_vectors) = (none_lists() for _ in range(4))
        midline_id_vectors, midline_length_vectors = (none_lists() for _ in range(2))
        (sarcomere_length_mean, sarcomere_length_std) = (nan_arrays() for _ in range(2))
        sarcomere_orientation_mean, sarcomere_orientation_std = nan_arrays(), nan_arrays()
        (n_vectors, n_vectors_removed, n_mbands, oop,
         sarcomere_area, sarcomere_area_ratio) = (nan_arrays() for _ in range(6))
        registry_lateral, registry_axial = none_lists(), none_lists()
        registry_lateral_mean, registry_axial_mean, registry_mean = (nan_arrays() for _ in range(3))

        # iterate images
        logger.info('Starting sarcomere length and orientation analysis...')
        for i, (frame_i, zbands_i, mbands_i, orientation_field_i, sarcomere_mask_i) in enumerate(
                progress_notifier.iterator(zip(list_frames, z_bands, mbands, orientation_field, sarcomere_mask),
                                           total=n_frames)):

            # Delegate to sarcomere_vectors module
            (
                pos_vectors_px_i, pos_vectors_i, midline_id_vectors_i, midline_length_vectors_i,
                sarcomere_length_vectors_i, sarcomere_orientation_vectors_i,
                n_mbands_i) = sarcomere_vectors.get_sarcomere_vectors(zbands_i, mbands_i,
                                                         orientation_field_i,
                                                         pixelsize=pixelsize,
                                                         median_filter_radius=median_filter_radius,
                                                         slen_lims=slen_lims,
                                                         interp_factor=interp_factor,
                                                         linewidth=linewidth,
                                                         interpolation_method=interpolation_method,
                                                         peak_prominence=peak_prominence,
                                                         threshold_mbands=threshold_mbands)

            # drop vectors that point across their neighbours (false positives at crossings)
            n_extracted_i = len(sarcomere_length_vectors_i)
            if max_neighbor_deviation is not None and n_extracted_i > 0:
                keep_i = sarcomere_vectors.filter_vectors_by_neighbors(
                    pos_vectors_i, sarcomere_orientation_vectors_i,
                    neighbor_radius=neighbor_radius, max_neighbor_deviation=max_neighbor_deviation)
                (pos_vectors_px_i, pos_vectors_i, midline_id_vectors_i, midline_length_vectors_i,
                 sarcomere_length_vectors_i, sarcomere_orientation_vectors_i) = (
                    arr[keep_i] for arr in (pos_vectors_px_i, pos_vectors_i, midline_id_vectors_i,
                                            midline_length_vectors_i, sarcomere_length_vectors_i,
                                            sarcomere_orientation_vectors_i))

            # Z-band registry of every vector (lateral: neighbours one myofibril over; axial: along the row)
            registry_lateral_i, registry_axial_i = sarcomere_vectors.zband_registry(
                pos_vectors_i, sarcomere_orientation_vectors_i, sarcomere_length_vectors_i,
                lateral_band=registry_lateral_band, axial_range=registry_axial_range,
                max_angle=registry_max_angle)
            registry_lateral[frame_i], registry_axial[frame_i] = registry_lateral_i, registry_axial_i
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)          # all-NaN frames give NaN, not a warning
                registry_lateral_mean[frame_i] = np.nanmean(registry_lateral_i) if registry_lateral_i.size else np.nan
                registry_axial_mean[frame_i] = np.nanmean(registry_axial_i) if registry_axial_i.size else np.nan
            registry_mean[frame_i] = 0.5 * (registry_lateral_mean[frame_i] + registry_axial_mean[frame_i])

            # write in list
            n_vectors[frame_i] = len(sarcomere_length_vectors_i)
            n_vectors_removed[frame_i] = n_extracted_i - len(sarcomere_length_vectors_i)
            n_mbands[frame_i] = n_mbands_i
            pos_vectors_px[frame_i] = pos_vectors_px_i
            pos_vectors[frame_i] = pos_vectors_i
            sarcomere_length_vectors[frame_i] = sarcomere_length_vectors_i
            sarcomere_orientation_vectors[frame_i] = sarcomere_orientation_vectors_i
            midline_id_vectors[frame_i] = midline_id_vectors_i
            midline_length_vectors[frame_i] = midline_length_vectors_i

            # calculate mean and std of sarcomere length and orientation
            sarcomere_length_mean[frame_i], sarcomere_length_std[frame_i], = np.nanmean(
                sarcomere_length_vectors_i), np.nanstd(sarcomere_length_vectors_i)
            # orientations are axial (θ and θ + π are the same axis): average the doubled
            # angles on the full circle and halve the result, so the mean lies in [0, π)
            orientations_i = np.asarray(sarcomere_orientation_vectors_i, dtype=float)   # a plain [] when no M-band
            orientations_i = orientations_i[~np.isnan(orientations_i)]
            if len(orientations_i) > 1:
                sarcomere_orientation_mean[frame_i] = stats.circmean(2 * orientations_i) / 2
                sarcomere_orientation_std[frame_i] = stats.circstd(2 * orientations_i) / 2

            # orientation order parameter
            if len(sarcomere_orientation_vectors_i) > 0:
                oop[frame_i], _ = Utils.analyze_orientations(
                    sarcomere_orientation_vectors_i[~np.isnan(sarcomere_orientation_vectors_i)])

            # Threshold before summing, so that this is a pixel count like
            # cell_mask_area, z_mask_area and domain_area, and sarcomere_area_ratio
            # divides two comparable quantities.
            sarcomere_area[frame_i] = (np.sum(sarcomere_mask_i > threshold_sarcomere_mask)
                                       * self.metadata.pixelsize ** 2)
            if 'structure.cell.mask_area' in self.data:
                sarcomere_area_ratio[frame_i] = sarcomere_area[frame_i] / self.data['structure.cell.mask_area'][frame_i]

        stripe_ratio = self._check_pixelsize(list_frames, pos_vectors_px, sarcomere_orientation_vectors,
                                             sarcomere_length_vectors)

        vectors_dict = {'params.structure.analyze_vectors.frames': list_frames,
                        'params.structure.analyze_vectors.threshold_sarcomere_mask': threshold_sarcomere_mask,
                        'params.structure.analyze_vectors.median_filter_radius': median_filter_radius,
                        'params.structure.analyze_vectors.slen_lims': slen_lims,
                        'params.structure.analyze_vectors.interp_factor': interp_factor,
                        'params.structure.analyze_vectors.interpolation_method': interpolation_method,
                        'params.structure.analyze_vectors.linewidth': linewidth,
                        'params.structure.analyze_vectors.smooth_orientation_sigma': smooth_orientation_sigma,
                        'params.structure.analyze_vectors.peak_prominence': peak_prominence,
                        'params.structure.analyze_vectors.neighbor_radius': neighbor_radius,
                        'params.structure.analyze_vectors.max_neighbor_deviation': max_neighbor_deviation,
                        'params.structure.analyze_vectors.registry_lateral_band': tuple(registry_lateral_band),
                        'params.structure.analyze_vectors.registry_axial_range': registry_axial_range,
                        'params.structure.analyze_vectors.registry_max_angle': registry_max_angle,
                        'params.structure.analyze_vectors.use_fast_movie_zbands': use_fast_movie_zbands,
                        'params.structure.analyze_vectors.zbands_source': zbands_source,
                        'structure.sarcomere.n_vectors': n_vectors,
                        'structure.sarcomere.n_vectors_removed': n_vectors_removed,
                        'structure.sarcomere.n_mbands': n_mbands, 'structure.sarcomere.pos_px': pos_vectors_px,
                        'structure.sarcomere.pos': pos_vectors, 'structure.sarcomere.slen': sarcomere_length_vectors,
                        'structure.sarcomere.orientation': sarcomere_orientation_vectors,
                        'structure.sarcomere.area': sarcomere_area, 'structure.sarcomere.area_ratio': sarcomere_area_ratio,
                        'structure.sarcomere.midline_length': midline_length_vectors, 'structure.sarcomere.midline_id': midline_id_vectors,
                        'structure.sarcomere.slen_mean': sarcomere_length_mean,
                        'structure.sarcomere.slen_std': sarcomere_length_std,
                        'structure.sarcomere.orientation_mean': sarcomere_orientation_mean,
                        'structure.sarcomere.orientation_std': sarcomere_orientation_std,
                        'structure.sarcomere.oop': oop,
                        'structure.sarcomere.registry_lateral': registry_lateral,
                        'structure.sarcomere.registry_axial': registry_axial,
                        'structure.sarcomere.registry_lateral_mean': registry_lateral_mean,
                        'structure.sarcomere.registry_axial_mean': registry_axial_mean,
                        'structure.sarcomere.registry_mean': registry_mean,
                        'structure.sarcomere.stripe_period_ratio': stripe_ratio}
        self.data.update(vectors_dict)
        if self.auto_save:
            self.commit()

    def _check_pixelsize(self, list_frames, pos_vectors_px, orientation_vectors, length_vectors,
                         min_vectors: int = 20) -> float:
        """Stripe period of the raw image / median sarcomere length on the first analyzed frame with at least
        ``min_vectors`` vectors; logs a warning when the pixel size looks too small
        (:func:`sarcomere_vectors.pixelsize_plausibility`). NaN when it cannot be measured."""
        frame = next((f for f in list_frames
                      if length_vectors[f] is not None and len(length_vectors[f]) >= min_vectors), None)
        if frame is None:
            return np.nan
        pixelsize = self.metadata.pixelsize
        slen_um = float(np.nanmedian(length_vectors[frame]))
        image = self._read_raw_frames([frame])[0]
        period_px = sarcomere_vectors.stripe_period(image, pos_vectors_px[frame], orientation_vectors[frame],
                                                    slen_um / pixelsize)
        message = sarcomere_vectors.pixelsize_plausibility(period_px, slen_um, pixelsize)
        if message:
            logger.warning(message)
        return float(period_px * pixelsize / slen_um) if np.isfinite(period_px) else np.nan

    def _vector_frames(self, frames) -> List[int]:
        """The frames of the sarcomere vectors that a step building on them (myofibrils, domains) analyzes:
        None or ``'all'`` = every frame analyze_vectors stored, otherwise the canonical ``frames``,
        which must all carry vectors."""
        analysed = sorted({int(f) for f in np.atleast_1d(
            self.data.get('params.structure.analyze_vectors.frames', [])).ravel()})
        requested = self._canonical_frames(frames)
        if requested == 'all':
            return analysed
        missing = [f for f in requested if f not in analysed]
        if missing:
            hint = ''
            detected = sorted(set(self._detected_frames() or []))
            if detected and analysed != detected and analysed == list(range(len(analysed))):
                hint = (f' After the partial detection of frames {detected}, analyze_vectors stored '
                        f'their vectors under positions {analysed}: pass frames={analysed}, or re-run '
                        f'analyze_vectors(frames={detected}) to store them under their frame numbers.')
            raise ValueError(f'Run analyze_vectors first for frames {missing} (sarcomere vectors '
                             f'are stored for frames {analysed}).{hint}')
        return requested

    def analyze_myofibrils(self, frames: Optional[Union[str, int, Sequence[int]]] = None,
                           ratio_seeds: float = 0.1, persistence: int = 3, threshold_distance: float = 0.5,
                           n_min: int = 4, median_filter_radius: float = 0.5, random_seed: Optional[int] = 42,
                           progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Estimate myofibril lines by line growth algorithm and analyze length and curvature.

        Parameters
        ----------
        frames : {'all', int, sequence of int} or None, optional
            Frames to analyze: a frame number, any sequence of frame numbers, or
            ``'all'`` / None for every frame that carries sarcomere vectors.
            Default is None.
        ratio_seeds : float, optional
            Ratio of sarcomere vectors used as seeds for line growth.
            Default is 0.1.
        persistence : int, optional
            Persistence of line (averaged vector length and orientation for prior
            estimation); must be > 0. Default is 3.
        threshold_distance : float, optional
            Maximal distance for nearest-neighbor estimation, in µm.
            Default is 0.5.
        n_min : int, optional
            Minimal number of sarcomere segments per line; shorter lines are
            removed. Default is 4.
        median_filter_radius : float, optional
            Filter radius for smoothing the myofibril length map, in µm.
            Default is 0.5.
        random_seed : int or None, optional
            Seed of the random choice of the start vectors, so the same image gives the same lines
            (and the same myofibril groups). None draws new start vectors on every run. Default is 42.
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in the GUI. Default is
            ProgressNotifier.progress_notifier_tqdm().
        """
        if 'structure.sarcomere.pos_px' not in self.data:
            raise ValueError('Sarcomere length and orientation not yet analyzed. Run analyze_vectors first.')
        list_frames = self._vector_frames(frames)

        pos_vectors_px = [self.data['structure.sarcomere.pos_px'][frame] for frame in list_frames]
        pos_vectors = [self.data['structure.sarcomere.pos'][frame] for frame in list_frames]
        sarcomere_length_vectors = [self.data['structure.sarcomere.slen'][frame] for frame in list_frames]
        sarcomere_orientation_vectors = [self.data['structure.sarcomere.orientation'][frame] for frame in list_frames]
        midline_length_vectors = [self.data['structure.sarcomere.midline_length'][frame] for frame in list_frames]

        # create empty arrays
        def none_lists():
            return [None] * self.metadata.n_stack
        def nan_arrays():
            return np.full(self.metadata.n_stack, np.nan)
        length_mean, length_std, length_max = (nan_arrays() for _ in range(3))
        bending_mean, bending_std = (nan_arrays() for _ in range(2))
        myof_lines, lengths, bending = (none_lists() for _ in range(3))

        # iterate frames
        logger.info('Starting myofibril line analysis...')
        for i, (
                frame_i, pos_vectors_px_i, pos_vectors_i, sarcomere_length_vectors_i, sarcomere_orientation_vectors_i,
                midline_length_vectors_i) in enumerate(
            progress_notifier.iterator(
                zip(list_frames, pos_vectors_px, pos_vectors, sarcomere_length_vectors, sarcomere_orientation_vectors,
                    midline_length_vectors),
                total=len(pos_vectors_px))):
            if pos_vectors_px_i is None:
                continue
            if len(np.asarray(pos_vectors_px_i).T) > 0:
                # Delegate to myofibril_analysis module
                line_data_i = myofibril_analysis.line_growth(pos_vectors_px_i, sarcomere_length_vectors_i,
                                               sarcomere_orientation_vectors_i,
                                               midline_length_vectors_t=midline_length_vectors_i,
                                               pixelsize=self.metadata.pixelsize, ratio_seeds=ratio_seeds,
                                               persistence=persistence, threshold_distance=threshold_distance,
                                               n_min=n_min, random_seed=random_seed)
                lines_i = line_data_i['lines']

                if len(lines_i) > 0:
                    # line lengths and mean squared curvature (msc)
                    lengths_i = line_data_i['line_features']['length_lines']
                    bending_i = line_data_i['line_features']['bending_lines']

                    if len(lengths_i) > 0:
                        # Delegate to myofibril_analysis module for map creation
                        myof_map_i = myofibril_analysis.create_myofibril_length_map(myof_lines=lines_i, myof_length=lengths_i,
                                                                      pos_vectors=pos_vectors_i,
                                                                      sarcomere_orientation_vectors=sarcomere_orientation_vectors_i,
                                                                      sarcomere_length_vectors=sarcomere_length_vectors_i,
                                                                      size=self.metadata.size,
                                                                      pixelsize=self.metadata.pixelsize,
                                                                      median_filter_radius=median_filter_radius)

                        myof_map_flat_i = myof_map_i.flatten()
                        myof_map_flat_i = myof_map_flat_i[~np.isnan(myof_map_flat_i)]
                        weights_i = 1.0 / myof_map_flat_i
                        weighted_mean_length_i = np.average(myof_map_flat_i, weights=weights_i)
                        weighted_std_length_i = np.sqrt(np.average((myof_map_flat_i - weighted_mean_length_i) ** 2,
                                                                   weights=weights_i))
                        length_mean[frame_i], length_std[frame_i], length_max[frame_i] = (weighted_mean_length_i,
                                                                                          weighted_std_length_i,
                                                                                          np.nanmax(myof_map_flat_i))
                        bending_mean[frame_i], bending_std[frame_i] = np.mean(bending_i), np.std(bending_i)
                    myof_lines[frame_i] = lines_i
                    lengths[frame_i] = lengths_i
                    bending[frame_i] = bending_i

        # update structure dictionary
        myofibril_data = {'structure.myofibril.length_mean': length_mean,
                          'structure.myofibril.length_std': length_std, 'structure.myofibril.lines': myof_lines,
                          'structure.myofibril.length_max': length_max, 'structure.myofibril.length': lengths,
                          'structure.myofibril.bending': bending,
                          'structure.myofibril.bending_mean': bending_mean,
                          'structure.myofibril.bending_std': bending_std,
                          'params.structure.analyze_myofibrils.persistence': persistence,
                          'params.structure.analyze_myofibrils.threshold_distance': threshold_distance,
                          'params.structure.analyze_myofibrils.frames': list_frames,
                          'params.structure.analyze_myofibrils.n_min': n_min,
                          'params.structure.analyze_myofibrils.ratio_seeds': ratio_seeds,
                          'params.structure.analyze_myofibrils.median_filter_radius': median_filter_radius,
                          'params.structure.analyze_myofibrils.random_seed': random_seed
                          }

        self.data.update(myofibril_data)
        if self.auto_save:
            self.commit()

    def analyze_domains(self, frames: Optional[Union[str, int, Sequence[int]]] = None,
                                  d_max: float = 3, cosine_min: float = 0.65, leiden_resolution: float = 0.06,
                                  random_seed: int = 42, area_min: float = 20.0, dilation_radius: float = 0.3,
                                  store_mask: bool = False,
                                  progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Cluster sarcomeres into domains based on their spatial and orientational properties using the Leiden algorithm
        for community detection.

        Parameters
        ----------
        frames : {'all', int, sequence of int} or None, optional
            Frames to analyze: a frame number, any sequence of frame numbers, or
            ``'all'`` / None for every frame that carries sarcomere vectors.
            Default is None.
        d_max : float, optional
            Max. distance threshold (µm) for creating a network edge between
            vector ends. Default is 3.
        cosine_min : float, optional
            Minimal absolute cosine between vector angles for creating a network
            edge between vector ends. Default is 0.65.
        leiden_resolution : float, optional
            Control parameter for domain size; smaller values favor larger
            domains, larger values favor smaller domains. Default is 0.06.
        random_seed : int, optional
            Random seed for the Leiden algorithm (reproducibility).
            Default is 42.
        area_min : float, optional
            Minimal area of domains/clusters in µm². Default is 20.0.
        dilation_radius : float, optional
            Dilation radius for refining domain area masks, in µm. Default is 0.3.
        store_mask : bool, optional
            If True, store the integer-labeled domain mask in ``self.data``
            (memory-intensive for large time-series). Default is False.
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in the GUI. Default is
            ProgressNotifier.progress_notifier_tqdm().
        """
        if 'structure.sarcomere.pos' not in self.data:
            raise ValueError('Sarcomere length and orientation not yet analyzed. Run analyze_vectors first.')
        list_frames = self._vector_frames(frames)

        pos_vectors = [np.asarray(self.data['structure.sarcomere.pos'][t]) for t in list_frames]
        sarcomere_length_vectors = [np.asarray(self.data['structure.sarcomere.slen'][t]) for t in list_frames]
        sarcomere_orientation_vectors = [np.asarray(self.data['structure.sarcomere.orientation'][t]) for t in list_frames]
        midline_id_vectors = [np.asarray(self.data['structure.sarcomere.midline_id'][t]) for t in list_frames]
        # Z-band registry per vector (absent in stores analysed before it existed: the domain registry is skipped)
        has_registry = 'structure.sarcomere.registry_lateral' in self.data
        if not has_registry:
            logger.info('No Z-band registry in the sarcomere vectors; re-run analyze_vectors for the '
                        'per-domain registry.')

        # create empty arrays
        def none_lists():
            return [None] * self.metadata.n_stack
        def nan_arrays():
            return np.full(self.metadata.n_stack, np.nan)
        n_domains, domain_area_mean, domain_area_std = (nan_arrays() for _ in range(3))
        domain_registry_lateral, domain_registry_axial = none_lists(), none_lists()
        domain_registry_lateral_mean, domain_registry_axial_mean = nan_arrays(), nan_arrays()
        domain_slen_mean, domain_slen_std = (nan_arrays() for _ in range(2))
        domain_oop_mean, domain_oop_std = (nan_arrays() for _ in range(2))

        (domains, domain_area, domain_slen, domain_slen_std,
         domain_oop, domain_orientation) = (none_lists() for _ in range(6))
        
        # optionally store domain masks
        if store_mask:
            domain_mask = none_lists()

        # iterate frames
        logger.info('Starting sarcomere domain analysis...')
        for i, (frame_i, pos_vectors_i, sarcomere_length_vectors_i, sarcomere_orientation_vectors_i,
                midline_id_vectors_i) in enumerate(
            progress_notifier.iterator(
                zip(list_frames, pos_vectors, sarcomere_length_vectors, sarcomere_orientation_vectors,
                    midline_id_vectors),
                total=len(pos_vectors))):
            if pos_vectors_i.ndim == 0:
                continue
            # Delegate to domain_clustering module
            cluster_data_t = domain_clustering.cluster_sarcomeres(pos_vectors_i, sarcomere_length_vectors_i,
                                                     sarcomere_orientation_vectors_i,
                                                     pixelsize=self.metadata.pixelsize,
                                                     size=self.metadata.size,
                                                     d_max=d_max, cosine_min=cosine_min,
                                                     leiden_resolution=leiden_resolution, random_seed=random_seed,
                                                     area_min=area_min, dilation_radius=dilation_radius)
            (n_domains[frame_i], domains[frame_i], domain_area[frame_i], domain_slen[frame_i], domain_slen_std[frame_i],
             domain_oop[frame_i], domain_orientation[frame_i], domain_mask_i) = cluster_data_t

            # optionally store domain mask as sparse matrix
            if store_mask:
                domain_mask[frame_i] = sparse.coo_matrix(domain_mask_i)

            # calculate mean and std of domains
            domain_area_mean[frame_i], domain_area_std[frame_i] = np.mean(domain_area[frame_i]), np.std(
                domain_area[frame_i])
            domain_slen_mean[frame_i], domain_slen_std[frame_i] = (
                np.mean(domain_slen[frame_i]), np.std(domain_slen[frame_i]))
            domain_oop_mean[frame_i], domain_oop_std[frame_i] = (
                np.mean(domain_oop[frame_i]), np.std(domain_oop[frame_i]))

            # Z-band registry per domain: mean over the member vectors that have a value
            if has_registry:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    reg_lat_all = np.asarray(self.data['structure.sarcomere.registry_lateral'][frame_i], float)
                    reg_ax_all = np.asarray(self.data['structure.sarcomere.registry_axial'][frame_i], float)
                    members = [np.asarray(idx, dtype=int) for idx in domains[frame_i]]
                    domain_registry_lateral[frame_i] = np.array(
                        [np.nanmean(reg_lat_all[idx]) if idx.size else np.nan for idx in members])
                    domain_registry_axial[frame_i] = np.array(
                        [np.nanmean(reg_ax_all[idx]) if idx.size else np.nan for idx in members])
                    domain_registry_lateral_mean[frame_i] = np.nanmean(domain_registry_lateral[frame_i]) \
                        if len(members) else np.nan
                    domain_registry_axial_mean[frame_i] = np.nanmean(domain_registry_axial[frame_i]) \
                        if len(members) else np.nan

        # update structure dictionary
        domain_data = {'structure.domain.n': n_domains, 'structure.domain.members': domains,
                       'structure.domain.area': domain_area, 'structure.domain.area_mean': domain_area_mean,
                       'structure.domain.area_std': domain_area_std,
                       'structure.domain.slen': domain_slen, 'structure.domain.slen_mean': domain_slen_mean,
                       'structure.domain.slen_std': domain_slen_std,
                       'structure.domain.oop': domain_oop, 'structure.domain.oop_mean': domain_oop_mean,
                       'structure.domain.oop_std': domain_oop_std,
                       'structure.domain.orientation': domain_orientation,
                       'params.structure.analyze_domains.frames': list_frames,
                       'params.structure.analyze_domains.d_max': d_max,
                       'params.structure.analyze_domains.cosine_min': cosine_min,
                       'params.structure.analyze_domains.leiden_resolution': leiden_resolution,
                       'params.structure.analyze_domains.area_min': area_min,
                       'params.structure.analyze_domains.dilation_radius': dilation_radius,
                       'params.structure.analyze_domains.store_mask': store_mask}
        
        if has_registry:
            domain_data.update({'structure.domain.registry_lateral': domain_registry_lateral,
                                'structure.domain.registry_axial': domain_registry_axial,
                                'structure.domain.registry_lateral_mean': domain_registry_lateral_mean,
                                'structure.domain.registry_axial_mean': domain_registry_axial_mean})

        # add domain mask if stored
        if store_mask:
            domain_data['structure.domain.mask'] = domain_mask

        self.data.update(domain_data)
        if self.auto_save:
            self.commit()

    def full_analysis(self, frames='all'):
        """
        Analyze sarcomere structure with default parameters at specified frames.

        Parameters
        ----------
        frames : {'all', int, sequence of int} or None, optional
            Frames to analyze ('all', a frame number, or any sequence of frame
            numbers). Default is 'all'.
        """
        frames = self._canonical_frames(frames)
        auto_save = self.auto_save
        self.auto_save = False                 # one store write at the end, not one per step
        try:
            self.analyze_z_bands(frames=frames)
            self.analyze_vectors(frames=frames)
            self.analyze_myofibrils(frames=frames)
            self.analyze_domains(frames=frames)
        finally:
            self.auto_save = auto_save
        self.commit()
