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


"""Helpers for working with optional values."""

from typing import TypeVar, Optional, cast, Callable


class TypeUtils:
    """Utility helpers for handling optional (possibly None) values."""

    T = TypeVar('T')

    @staticmethod
    def unbox(optional: Optional[T]) -> T:
        """
        Return the value of an optional, raising ValueError if it is None.

        Parameters
        ----------
        optional : T or None
            Value to unbox.

        Returns
        -------
        T
            The unboxed value.
        """
        if optional is None:
            raise ValueError('Expected a value, got None (is a file open?).')
        return cast(TypeUtils.T, optional)

    @staticmethod
    def if_present(optional: Optional[T], callback: Callable[[T], None]) -> None:
        """
        Invoke a callback with the value only if it is not None.

        Parameters
        ----------
        optional : T or None
            Value to test and pass to the callback.
        callback : callable
            Function called with the unboxed value when ``optional`` is not None.
        """
        if optional is not None:
            callback(TypeUtils.unbox(optional))

