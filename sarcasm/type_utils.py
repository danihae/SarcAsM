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


from typing import TypeVar, Optional, cast, Callable


class TypeUtils:
    T = TypeVar('T')

    @staticmethod
    def unbox(optional: Optional[T], throw_exception=True) -> T:
        if optional is None and throw_exception:
            raise ValueError('Variable of type' + type(optional).__name__ + ' is None')
        return cast(TypeUtils.T, optional)
        pass

    @staticmethod
    def if_present(optional: Optional[T], callback: Callable[[T], None]) -> None:
        if optional is not None:
            callback(TypeUtils.unbox(optional))
        pass

    pass
