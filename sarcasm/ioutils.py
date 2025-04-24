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


import copy
import json

import numpy as np
import orjson
from scipy import sparse


class IOUtils:
    """Utility functions for storing and loading IO data."""

    @staticmethod
    def __serialize_field(field):
        if sparse.issparse(field):
            return {
                'type': 'sparse_matrix',
                'values': IOUtils.__sparse_to_json_serializable(field)
            }
        elif isinstance(field, np.ndarray):
            return {'type': 'ndarray', 'values': field.tolist()}
        elif isinstance(field, list):
            return [IOUtils.__serialize_field(val) for val in field]
        elif isinstance(field, dict):
            return {key: IOUtils.__serialize_field(value) for key, value in field.items()}
        elif isinstance(field, np.generic):
            return {'value': field.item(), 'type': field.dtype.name}
        else:
            return field

    @staticmethod
    def __deserialize_field(field):
        if isinstance(field, list):
            return [IOUtils.__deserialize_field(val) for val in field]
        elif isinstance(field, dict) and 'type' in field:
            if field['type'] == 'ndarray':
                return np.array(field['values'])
            elif field['type'] == 'sparse_matrix':
                return IOUtils.__json_serializable_to_sparse(field['values'])
            else:
                dtype = np.dtype(field['type'])
                return np.array(field['value'], dtype=dtype)
        elif isinstance(field, dict):
            return {key: IOUtils.__deserialize_field(value) for key, value in field.items()}
        else:
            return field

    @staticmethod
    def json_serialize(obj, file_path):
        cpy = copy.deepcopy(obj)
        cpy = IOUtils.__serialize_field(cpy)
        try:
            # Write as binary using orjson to boost performance.
            with open(file_path, 'wb') as f:
                f.write(orjson.dumps(
                    cpy,
                    option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2
                ))
        except Exception as e:
            raise Exception(f"JSON serialization failed: {e}")

    @staticmethod
    def json_deserialize(file_path):
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return IOUtils.__deserialize_field(orjson.loads(content))
        except Exception as e:
            # Fallback using standard json (less strict about malformed JSON)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content_text = f.read()
                data = json.loads(content_text)
                return IOUtils.__deserialize_field(data)
            except Exception as fallback_error:
                raise Exception(
                    f"JSON deserialization failed for {file_path}: {e}. Fallback failed: {fallback_error}"
                )

    @staticmethod
    def __sparse_to_json_serializable(sparse_matrix):
        sparse_coo = sparse_matrix.tocoo()
        serializable_data = {
            "data": sparse_coo.data.tolist(),
            "row": sparse_coo.row.tolist(),
            "col": sparse_coo.col.tolist(),
            "shape": sparse_coo.shape
        }
        return orjson.dumps(serializable_data).decode('utf-8')

    @staticmethod
    def __json_serializable_to_sparse(json_data):
        data = orjson.loads(json_data.encode('utf-8'))
        return sparse.coo_matrix(
            (np.array(data["data"]),
             (np.array(data["row"]),
              np.array(data["col"]))),
            shape=tuple(data["shape"])
        )
