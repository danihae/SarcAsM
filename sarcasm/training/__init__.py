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



"""Tools to make training data and (re)train the networks.

* :mod:`~sarcasm.training.data_generation` -- targets for the 2D U-Net from images
  (:class:`~sarcasm.training.data_generation.TrainingDataGenerator`, see the training-data tutorial)
* :mod:`~sarcasm.training.synthetic_motion` -- synthetic movies from 2D training images
* :mod:`~sarcasm.training.teacher_labels` -- labelled clips from real movies via the 2D model
* :mod:`~sarcasm.training.unet3d_trainer` -- the 3D U-Net training loop

The step-by-step recipe for the 3D U-Net is in ``scripts/retrain_unet3d/README.md``.
"""
