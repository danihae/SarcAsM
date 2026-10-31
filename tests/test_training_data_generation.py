# -*- coding: utf-8 -*-
"""``TrainingDataGenerator`` runs end to end on the 1.0 API (SARC-26).

Its ``create_sarcomere_mask`` called a ``SarcAsM.sarcomere_mask`` that has not existed since
0.5.1, and nothing tested the class. This runs the tutorial's sequence on a crop of a real
image and checks the sarcomere mask against the bundled model's own sarcomere mask, which
would catch an orientation-convention error (a rotated or mirrored mask overlaps it far less).
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile
import torch

from sarcasm import TrainingDataGenerator, Utils
from sarcasm.analysis.detection import detect_sarcomeres_unet


@pytest.fixture
def crop_path(structure_single_image_path, tmp_path):
    with tifffile.TiffFile(structure_single_image_path) as tif:
        image = tif.asarray()[:256, :256]
        x_res = tif.pages[0].tags['XResolution'].value
    out = tmp_path / 'image' / 'crop.tif'
    out.parent.mkdir()
    tifffile.imwrite(out, image, imagej=True, resolution=(x_res[0] / x_res[1], x_res[0] / x_res[1]),
                     metadata={'unit': 'um'})
    return str(out)


def test_tutorial_sequence_produces_all_targets(crop_path, tmp_path):
    dirs = {name: str(tmp_path / name) + '/' for name in
            ('zbands', 'zbands_prelim', 'mbands', 'orientation', 'sarcomere_mask')}
    for d in dirs.values():
        (tmp_path / d).mkdir(exist_ok=True)
    gen = TrainingDataGenerator(image_path=crop_path, output_dirs=dirs)
    assert gen.pixelsize == pytest.approx(0.1075, abs=1e-3)

    gen.predict_zbands(patch_size=(256, 256))               # bundled generalist model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    gen.wavelet_analysis(device=device, score_threshold=0.25, minor=0.2, major=1,
                         len_lims=(1.4, 2.7), len_step=0.05, orient_step=5)
    assert len(gen.wavelet_dict['pos_vectors_px']) > 100
    gen.create_sarcomere_mask(dilation_radius=0.2)
    gen.create_orientation_map()
    gen.smooth_orientation_map(window_size=5)

    assert gen.zbands.shape == gen.mbands.shape == gen.sarcomere_mask.shape == (256, 256)
    assert np.isfinite(gen.orientation).any()
    reference = detect_sarcomeres_unet(gen.image[None], None, Utils.get_models_dir(), gen.pixelsize,
                                       max_patch_size=(256, 256))['sarcomere_mask'][0] > 0.5
    mask = gen.sarcomere_mask.astype(bool)
    iou = (mask & reference).sum() / (mask | reference).sum()
    assert iou > 0.6, f'sarcomere mask overlaps the model mask by IoU {iou:.2f} only'
