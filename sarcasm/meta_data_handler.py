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

import datetime
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict

import numpy as np

from sarcasm._version import __version__


@dataclass
class ImageMetadata:
    """Metadata of tif file."""

    # Core image properties (set during read_imgs)
    axes: str | None = None
    pixelsize: Optional[float] = None
    frametime: Optional[float] = None
    shape_orig: Tuple[int, ...] = field(default_factory=tuple)
    shape: Tuple[int, ...] | None = None
    n_stack: int | None = None
    size: Tuple[int, int] | None = None
    timestamps: Optional[List[float]] = None

    # File properties (set during initialization)
    file_name: str = ""
    file_path: str = ""

    # Computed properties (set in __post_init__) and SarcAsM metadata
    time: Optional[np.ndarray] = field(init=False, repr=False)
    sarcasm_version: str = field(default_factory=lambda: __version__)
    timestamp_analysis: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    # User-specified channel with sarcomere signal
    channel: Optional[int] = None

    # User-supplied metadata (dynamic)
    user_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived fields after initialization."""
        if not hasattr(self, 'sarcasm_version') or self.sarcasm_version is None:
            self.sarcasm_version = __version__
        if not hasattr(self, 'timestamp_analysis') or self.timestamp_analysis is None:
            self.timestamp_analysis = datetime.datetime.now().isoformat()

        # Create time array if we have both frametime and a stack
        if self.frametime is not None and self.n_stack is not None and self.n_stack > 1:
            self.time = np.arange(0, self.n_stack * self.frametime, self.frametime)
        else:
            self.time = None

    def add_user_info(self, **kwargs):
        """Add arbitrary user metadata after initialization."""
        self.user_info.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = asdict(self)

        # Flatten user_info into the main dict
        user_info = result.pop('user_info', {})
        result.update(user_info)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageMetadata':
        """Create from dictionary (for loading from JSON)."""
        # Get only fields that can be passed to __init__ (init=True)
        dataclass_fields = cls.__dataclass_fields__
        init_fields = {name for name, field_obj in dataclass_fields.items() if field_obj.init}

        # Separate init fields from user info, excluding init=False fields
        known_data = {k: v for k, v in data.items() if k in init_fields}
        user_data = {k: v for k, v in data.items() if k not in init_fields}

        # Create instance
        instance = cls(**known_data)

        # Add remaining data as user_info
        remaining_user_data = {k: v for k, v in user_data.items()
                               if k not in dataclass_fields}
        instance.add_user_info(**remaining_user_data)

        return instance

    @classmethod
    def save_to_file(cls, instance, filepath: Path):
        """Save metadata to JSON file."""
        # Convert numpy array to list for JSON serialization
        data = instance.to_dict()
        if 'time' in data and isinstance(data['time'], np.ndarray):
            data['time'] = data['time'].tolist()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Path) -> 'ImageMetadata':
        """Load metadata from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Convert time list back to numpy array
        if 'time' in data and isinstance(data['time'], list):
            data['time'] = np.array(data['time'])
        return cls.from_dict(data)
