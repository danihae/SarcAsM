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


"""Base class of the analysis namespaces ``sarc.structure`` and ``sarc.motion``."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sarcasm.core import SarcAsM


class _Namespace:
    """One analysis namespace of a recording (``sarc.structure`` or ``sarc.motion``).

    It holds no state of its own: results, metadata and the store belong to the
    :class:`~sarcasm.core.SarcAsM` recording, which the namespace reads and writes
    through the properties below.
    """

    def __init__(self, sarc: "SarcAsM") -> None:
        self._sarc = sarc

    def __repr__(self) -> str:
        return f"<{type(self).__name__} of {self._sarc!r}>"

    # -- the recording's state ---------------------------------------------- #
    @property
    def data(self):
        return self._sarc.data

    @property
    def metadata(self):
        return self._sarc.metadata

    @property
    def file_path(self):
        return self._sarc.file_path

    @property
    def store(self):
        return self._sarc.store

    @property
    def device(self):
        return self._sarc.device

    @property
    def model_dir(self):
        return self._sarc.model_dir

    @property
    def auto_save(self):
        return self._sarc.auto_save

    @auto_save.setter
    def auto_save(self, value):
        self._sarc.auto_save = value

    # -- the recording's helpers -------------------------------------------- #
    def read_imgs(self, *args, **kwargs):
        return self._sarc.read_imgs(*args, **kwargs)

    def commit(self):
        return self._sarc.commit()

    def read_mask(self, *args, **kwargs):
        return self._sarc.read_mask(*args, **kwargs)

    def mask_exists(self, *args, **kwargs):
        return self._sarc.mask_exists(*args, **kwargs)

    def _canonical_frame(self, *args, **kwargs):
        return self._sarc._canonical_frame(*args, **kwargs)

    def _canonical_frames(self, *args, **kwargs):
        return self._sarc._canonical_frames(*args, **kwargs)

    def _detected_frames(self):
        return self._sarc._detected_frames()

    def _predict_on_gpu(self, *args, **kwargs):
        return self._sarc._predict_on_gpu(*args, **kwargs)
