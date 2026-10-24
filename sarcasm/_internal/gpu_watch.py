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

"""Detect GPU work that macOS aborted during a prediction on Apple GPUs (MPS).

When the display needs the GPU, macOS may cancel a long-running Metal command buffer
(``kIOGPUCommandBufferCallbackErrorImpactingInteractivity``). PyTorch raises nothing:
Metal prints "command buffer exited with error status" to the process's stderr and
the affected output tensors hold garbage, which would be stored as masks. The only
signal is that message, so :func:`watch_metal_errors` captures the stderr file
descriptor while a prediction runs, passes everything through afterwards and reports
whether the message appeared.
"""

import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass

#: what Metal prints when a command buffer was aborted
METAL_ABORT_MARKERS = (b'command buffer exited with error status',
                       b'kIOGPUCommandBufferCallbackError')


@dataclass
class MetalWatch:
    failed: bool = False


@contextmanager
def watch_metal_errors(device):
    """Yield a :class:`MetalWatch` whose ``failed`` is True after the block if Metal reported
    an aborted command buffer. A no-op (``failed`` stays False) on devices other than MPS."""
    watch = MetalWatch()
    if getattr(device, 'type', str(device)) != 'mps':
        yield watch
        return
    try:
        sys.stderr.flush()
    except Exception:
        pass
    saved = os.dup(2)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield watch
        finally:
            try:
                sys.stderr.flush()
            except Exception:
                pass
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            captured = tmp.read()
            if captured:
                os.write(2, captured)          # nothing printed during the prediction is lost
            watch.failed = any(m in captured for m in METAL_ABORT_MARKERS)
