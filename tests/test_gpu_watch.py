# -*- coding: utf-8 -*-
"""Aborted Metal command buffers (macOS reclaims the GPU for the display) must not end up as masks."""
import os

import pytest
import torch

from sarcasm._internal.gpu_watch import watch_metal_errors
from sarcasm.core import SarcAsM

MPS = torch.device('mps')
ABORT = b'Error: command buffer exited with error status.\n'


def test_detects_the_metal_abort_message_and_passes_stderr_through(capfd):
    with watch_metal_errors(MPS) as watch:
        os.write(2, b'progress 50%\n' + ABORT)
    assert watch.failed
    assert 'progress 50%' in capfd.readouterr().err        # nothing printed meanwhile is lost


def test_clean_run_and_other_devices():
    with watch_metal_errors(MPS) as watch:
        os.write(2, b'all fine\n')
    assert not watch.failed
    with watch_metal_errors(torch.device('cpu')) as watch:
        os.write(2, ABORT)                                  # not captured off MPS
    assert not watch.failed


def _fake(device):
    sarc = SarcAsM.__new__(SarcAsM)
    sarc.device = device
    return sarc


def test_prediction_is_repeated_once_after_an_abort():
    calls = []

    def predict():
        calls.append(1)
        if len(calls) == 1:
            os.write(2, ABORT)

    _fake(MPS)._predict_on_gpu(predict, 'detect_sarcomeres')
    assert len(calls) == 2


def test_two_aborts_raise_instead_of_storing_wrong_masks():
    with pytest.raises(RuntimeError, match='cannot be trusted'):
        _fake(MPS)._predict_on_gpu(lambda: os.write(2, ABORT), 'detect_sarcomeres')
