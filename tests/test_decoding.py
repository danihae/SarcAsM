"""Tests for the shared decoder: one rule from probability to contraction intervals.

SarcAsM and CardioTrace both call :func:`decode_contractions`, so the same probabilities and
settings give them the same intervals. These tests pin the rules, and check that each tool's
earlier mask clean-up is the decoder with its own settings, without the gap closing and
with ``round()``.
"""

from __future__ import annotations

import functools
import os

import numpy as np
import pytest
from scipy import ndimage

from contraction_net.benchmark import postprocess
from contraction_net.contraction_net import ContractionNet
from contraction_net.decoding import (DEFAULT_PEAK, decode_contractions, intervals_to_mask,
                                      recommended_decoding)
from contraction_net.prediction import predict_contractions
from tests.test_polarity import regression_set

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_MODEL = os.path.join(REPO, 'sarcasm', 'models', 'model_ContractionNet.pt')

#: Frametimes of the data the tools see: CardioTrace at 30 fps, the EHT video, SarcAsM's
#: 61-fps recordings and the corpus.
FRAMETIMES = (1 / 30, 0.03336666666666588, 0.0164, 1 / 61)


def _decode(prob, frametime=1 / 30, threshold=0.5, peak=0.0, min_duration_s=0.0,
            border_frames=0):
    return decode_contractions(prob, frametime, threshold=threshold, peak=peak,
                               min_duration_s=min_duration_s, border_frames=border_frames)


def _spans(mask):
    labelled, _ = ndimage.label(mask)
    return [(int(s.start), int(s.stop)) for (s,) in ndimage.find_objects(labelled)]


def _cardiotrace_cleanup(prob, threshold, frametime, min_len, buffer_size=5):
    """CardioTrace's mask clean-up before the shared decoder: a threshold, then
    ``clear_border_scipy`` (any frame within ``buffer_size`` of an end) and
    ``remove_small_objects_scipy`` (fewer than ``round(min_len * framerate)`` frames)."""
    state = np.asarray(prob) > threshold
    labelled, _ = ndimage.label(state)
    border = np.zeros(state.size, bool)
    border[:buffer_size] = True
    border[state.size - buffer_size:state.size] = True
    for lab in np.unique(labelled[border]):
        if lab:
            state[labelled == lab] = False
    labelled, _ = ndimage.label(state)
    framerate = 1 / frametime
    small = np.bincount(labelled.ravel()) < int(np.round(min_len * framerate))
    small[0] = False
    state[small[labelled]] = False
    return _spans(state)


def _sarcasm_cleanup(prob, threshold, frametime, contr_time_min):
    """SarcAsM's mask clean-up without its gap closing and with ``round()``: an opening by
    the minimum duration, which removes shorter runs and keeps the rest whole."""
    opening = np.ones(max(1, int(round(contr_time_min / frametime))))
    return _spans(ndimage.binary_opening(np.asarray(prob) > threshold, structure=opening))


@functools.lru_cache(maxsize=None)
def _probabilities():
    """Probability traces to decode: smoothed noise, plus the bundled model on the regression
    set."""
    rng = np.random.default_rng(12)
    out = [ndimage.uniform_filter1d(rng.random(int(rng.integers(50, 3000))), int(k))
           for k in rng.integers(1, 9, 60)]
    for _, x, ft in regression_set():
        prob = predict_contractions(x, SHIPPED_MODEL, frametime=ft)[0]
        if prob.any():
            out.append(prob)
    return tuple(out)


class TestRules:
    def test_runs_are_strictly_above_the_threshold(self):
        prob = np.array([0, .6, .6, .6, .5, .9, .9, .9, 0])
        assert _decode(prob, threshold=0.5) == [(1, 4), (5, 8)]

    def test_a_run_must_reach_the_peak(self):
        prob = np.array([0, .7, .85, .7, 0, .7, .84, .7, 0])
        assert _decode(prob, threshold=0.6, peak=0.85) == [(1, 4)]

    def test_minimum_duration_is_rounded(self):
        # 0.2 s at the video's 33.37 ms is 5.99 frames: round() keeps 6-frame runs and
        # drops 5-frame ones, where int() would have kept both
        prob = np.zeros(40)
        prob[2:7], prob[10:16] = 0.9, 0.9
        assert _decode(prob, 0.03336666666666588, min_duration_s=0.2) == [(10, 16)]
        # halves go to the even frame count: 0.15 s at 30 fps is 4.5, so 4 frames
        prob = np.zeros(40)
        prob[2:6], prob[10:13] = 0.9, 0.9
        assert _decode(prob, 1 / 30, min_duration_s=0.15) == [(2, 6)]
        # and never less than one frame
        prob = np.zeros(10)
        prob[4] = 0.9
        assert _decode(prob, 1 / 30, min_duration_s=0.0) == [(4, 5)]

    def test_runs_near_either_end_are_dropped(self):
        prob = np.zeros(40)
        for start, stop in ((4, 8), (12, 16), (31, 35)):
            prob[start:stop] = 0.9
        assert _decode(prob, border_frames=5) == [(12, 16), (31, 35)]
        prob[35] = 0.9   # the run now reaches frame 35, the first of the last five
        assert _decode(prob, border_frames=5) == [(12, 16)]

    def test_border_zero_keeps_runs_touching_the_ends(self):
        prob = np.zeros(20)
        prob[:3], prob[-3:] = 0.9, 0.9
        assert _decode(prob) == [(0, 3), (17, 20)]

    def test_no_gap_is_closed(self):
        # the one-frame hole the per-twitch label puts between overlapping twitches
        prob = np.r_[np.full(10, .9), .1, np.full(10, .9)]
        assert _decode(prob, threshold=0.5, peak=0.85) == [(0, 10), (11, 21)]

    @pytest.mark.parametrize('length', [800, 1600, 3000])
    def test_a_sustained_contraction_is_one_event(self, length):
        prob = np.r_[np.zeros(400), np.full(length, 0.95), np.zeros(400)]
        assert decode_contractions(prob, 1 / 30, threshold=0.6, peak=0.85, min_duration_s=0.1,
                                   border_frames=5) == [(400, 400 + length)]

    @pytest.mark.parametrize('threshold', [0.0, 0.3, 0.6])
    def test_an_abstained_trace_decodes_to_nothing(self, threshold):
        assert _decode(np.zeros((3, 500), np.float32)[0], threshold=threshold) == []

    def test_mask_of_the_intervals(self):
        prob = np.r_[0, .9, .9, 0, 0, .9, 0]
        mask = intervals_to_mask(_decode(prob), prob.size)
        assert mask.dtype == bool and np.array_equal(mask, prob > 0.5)
        assert not intervals_to_mask([], 5).any()

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError, match='frametime'):
            _decode(np.zeros(10), frametime=0.0)
        with pytest.raises(ValueError, match='border_frames'):
            _decode(np.zeros(10), border_frames=-1)

    @pytest.mark.parametrize('shape', [(3, 100), (1, 100), ()])
    def test_only_one_channel_is_decoded(self, shape):
        # the whole output of predict_contractions would otherwise be decoded channel after
        # channel, with intervals beyond the end of the trace
        with pytest.raises(ValueError, match='1D'):
            _decode(np.full(shape, 0.9))

    def test_recommended_decoding_falls_back_for_bare_modules(self):
        assert recommended_decoding(ContractionNet(n_filter=8)) == {'threshold': 0.5,
                                                                    'peak': DEFAULT_PEAK}


class TestSharedDecoder:
    """Each tool's earlier clean-up is the shared decoder at that tool's settings, with no
    confidence peak, so both read the same intervals from the same probabilities."""

    @pytest.mark.parametrize('frametime', FRAMETIMES)
    def test_cardiotrace_settings(self, frametime):
        for prob in _probabilities():
            for min_len in (0.1, 0.15):
                assert (_decode(prob, frametime, threshold=0.6, min_duration_s=min_len,
                                border_frames=5)
                        == _cardiotrace_cleanup(prob, 0.6, frametime, min_len))

    @pytest.mark.parametrize('frametime', FRAMETIMES)
    def test_sarcasm_settings(self, frametime):
        for prob in _probabilities():
            assert (_decode(prob, frametime, threshold=0.6, min_duration_s=0.2)
                    == _sarcasm_cleanup(prob, 0.6, frametime, 0.2))

    def test_the_same_probabilities_give_the_same_intervals(self):
        # nothing but the probabilities and the settings goes in, whichever tool calls
        for prob in _probabilities()[:20]:
            first = decode_contractions(prob, 0.0164, threshold=0.6, peak=0.85,
                                        min_duration_s=0.2)
            again = decode_contractions(prob.astype(np.float64).tolist(), 0.0164,
                                        threshold=0.6, peak=0.85, min_duration_s=0.2)
            assert first == again


class TestBenchmarkPostprocess:
    """benchmark.postprocess no longer closes gaps by default."""

    def test_a_one_frame_gap_stays_open(self):
        mask = np.r_[np.ones(10, bool), False, np.ones(10, bool)]
        assert _spans(postprocess(mask)) == [(0, 10), (11, 21)]

    def test_an_explicit_closing_still_closes(self):
        # kept only to reproduce results made with a closing, such as the corpus labels
        mask = np.r_[np.zeros(5, bool), np.ones(10, bool), False, np.ones(10, bool),
                     np.zeros(5, bool)]
        assert _spans(postprocess(mask, merge_frames=2)) == [(5, 26)]
        assert _spans(postprocess(mask, frametime=0.0164, contr_time_min=0.03,
                                  merge_time_max=0.05)) == [(5, 26)]

    def test_short_runs_are_still_removed(self):
        mask = np.r_[False, True, True, False, np.ones(5, bool), False]
        assert _spans(postprocess(mask, min_frames=3)) == [(4, 9)]
