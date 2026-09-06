"""Real-data check of the image-flow motion predictor on the high-frame-rate reference movie.

At this frame rate a sarcomere moves ~1 px per frame, well inside the gates, so the
predictor must reproduce the hold-position tracker: nearly identical links, identical
lengths on identical links, no loss of continuity. (Its benefit is at coarse frame
rates, covered by the synthetic-GT test.)

Nothing is written to the reference store: the module-level tracker is fed the
stored vectors directly and the SarcAsM object is opened with ``auto_save=False``.
"""
from __future__ import annotations

import numpy as np
import pytest

from sarcasm import SarcAsM
from sarcasm.analysis import sarcomere_tracking as stk
from sarcasm.analysis.optical_flow import ImageFlowPredictor

N_FRAMES = 200


def _links(out, T):
    D = out['motion.tracks.detection_id']
    L = set()
    for t in range(T - 1):
        a, b = D[:, t], D[:, t + 1]
        m = (a >= 0) & (b >= 0)
        L.update(zip([t] * int(m.sum()), a[m].tolist(), b[m].tolist()))
    return L


def _vectors(sarc, frames):
    d = sarc.data
    def col(key, dtype, empty):
        return [np.asarray(d[key][t], dtype) if d[key][t] is not None else np.zeros(empty, dtype) for t in frames]
    return (col('structure.sarcomere.pos_px', np.int32, (0, 2)), col('structure.sarcomere.midline_id', np.int64, 0),
            col('structure.sarcomere.slen', np.float32, 0), col('structure.sarcomere.orientation', np.float32, 0))


def test_flow_predictor_is_neutral_on_the_61fps_reference_movie(motion_file_path):
    sarc = SarcAsM(motion_file_path, auto_save=False)
    if 'structure.sarcomere.pos_px' not in sarc.data:
        pytest.skip('reference store has no sarcomere vectors')
    frames = list(range(N_FRAMES))
    pos, mid, sl, ori = _vectors(sarc, frames)
    px, ft = sarc.metadata.pixelsize, sarc.metadata.frametime
    base = stk.track_sarcomere_vectors(pos, mid, sl, ori, pixelsize=px, frametime=ft)
    flow = stk.track_sarcomere_vectors(pos, mid, sl, ori, pixelsize=px, frametime=ft,
                                       predictor=ImageFlowPredictor(lambda k: sarc.read_imgs(frames=frames[k])))
    LB, LF = _links(base, N_FRAMES), _links(flow, N_FRAMES)
    assert len(LB & LF) / len(LB) >= 0.97
    assert abs(flow['motion.tracks.fragmentation_ratio'] - base['motion.tracks.fragmentation_ratio']) < 0.03
    cov_b = np.median(base['motion.tracks.observed'].mean(1)); cov_f = np.median(flow['motion.tracks.observed'].mean(1))
    assert cov_f >= cov_b - 0.02


def test_wrapper_rejects_an_unknown_predictor(motion_file_path):
    sarc = SarcAsM(motion_file_path, auto_save=False)
    with pytest.raises(ValueError, match='motion_predictor'):
        sarc.track_sarcomere_vectors(motion_predictor='bogus')
