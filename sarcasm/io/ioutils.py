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

"""JSON (de)serialization of SarcAsM results (``IOUtils``).

Round-trips nested results dicts to/from the ``structure.json`` format,
preserving numpy arrays, scalars and scipy sparse matrices.
"""

import copy
import json
import logging

import numpy as np
import orjson
from scipy import sparse

logger = logging.getLogger(__name__)


class IOUtils:
    """Utility functions for storing and loading IO data."""

    @staticmethod
    def __serialize_field(field):
        """Recursively convert a value into JSON-serializable form.

        Parameters
        ----------
        field : Any
            Value to encode (ndarray, sparse matrix, np.generic, list, dict,
            or plain JSON type).

        Returns
        -------
        Any
            JSON-serializable representation, tagging arrays/sparse matrices/
            scalars with a ``'type'`` field for round-trip.
        """
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
        """Recursively decode a value produced by :meth:`__serialize_field`.

        Parameters
        ----------
        field : Any
            Encoded value (possibly a type-tagged dict for ndarray, sparse
            matrix or scalar), list, dict, or plain JSON type.

        Returns
        -------
        Any
            Reconstructed value, restoring numpy arrays (NaN-aware), sparse
            matrices and scalars.
        """
        if isinstance(field, list):
            return [IOUtils.__deserialize_field(val) for val in field]
        elif isinstance(field, dict) and 'type' in field:
            if field['type'] == 'ndarray':
                arr = np.array(field['values'])
                # orjson serialises float NaN as JSON null; those round-trip
                # as Python None and force np.array into object dtype. If the
                # remaining non-None elements are all numeric, coerce back to
                # float with NaN in the None slots.
                if arr.dtype == object:
                    try:
                        arr = np.where(arr == None, np.nan, arr).astype(np.float64)  # noqa: E711
                    except (TypeError, ValueError):
                        pass  # genuinely non-numeric object array — leave it
                return arr
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
        """Serialize a results object to a JSON file.

        Parameters
        ----------
        obj : Any
            Object to serialize (typically a nested results dict). Deep-copied
            before encoding.
        file_path : str
            Destination path. Written as binary via ``orjson`` with sorted
            keys and 2-space indent.

        Raises
        ------
        Exception
            If serialization fails.
        """
        cpy = copy.deepcopy(obj)
        cpy = IOUtils.__serialize_field(cpy)
        try:
            # Write as binary using orjson to boost performance.
            with open(file_path, 'wb') as f:
                f.write(orjson.dumps(
                    cpy,
                    option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2
                ))
            logger.debug(f"Successfully serialized data to {file_path}")
        except Exception as e:
            logger.error(f"JSON serialization failed for {file_path}: {e}")
            raise Exception(f"JSON serialization failed: {e}") from e

    @staticmethod
    def json_deserialize(file_path):
        """Load and decode a results object from a JSON file.

        Parameters
        ----------
        file_path : str
            Path to the JSON file. Read with ``orjson``; on failure falls back
            to the standard ``json`` parser.

        Returns
        -------
        Any
            Deserialized object with numpy arrays, scalars and sparse matrices
            reconstructed.

        Raises
        ------
        Exception
            If both the primary and fallback parsers fail.
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            logger.debug(f"Successfully deserialized data from {file_path}")
            return IOUtils.__deserialize_field(orjson.loads(content))
        except Exception as e:
            logger.warning(f"orjson deserialization failed for {file_path}: {e}. Attempting fallback with standard json...")
            # Fallback using standard json (less strict about malformed JSON)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content_text = f.read()
                data = json.loads(content_text)
                logger.info(f"Successfully recovered data from {file_path} using fallback json parser")
                return IOUtils.__deserialize_field(data)
            except Exception as fallback_error:
                logger.error(f"JSON deserialization failed for {file_path}. Primary error: {e}. Fallback error: {fallback_error}")
                raise Exception(
                    f"JSON deserialization failed for {file_path}: {e}. Fallback failed: {fallback_error}"
                ) from fallback_error

    @staticmethod
    def __sparse_to_json_serializable(sparse_matrix):
        """Encode a sparse matrix as a JSON string of COO components.

        Parameters
        ----------
        sparse_matrix : scipy.sparse.spmatrix
            Matrix to encode.

        Returns
        -------
        str
            JSON string with ``data``, ``row``, ``col`` and ``shape`` of the
            COO representation.
        """
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
        """Decode a JSON string of COO components back into a sparse matrix.

        Parameters
        ----------
        json_data : str
            JSON string produced by :meth:`__sparse_to_json_serializable`.

        Returns
        -------
        scipy.sparse.coo_matrix
            Reconstructed sparse matrix.
        """
        data = orjson.loads(json_data.encode('utf-8'))
        return sparse.coo_matrix(
            (np.array(data["data"]),
             (np.array(data["row"]),
              np.array(data["col"]))),
            shape=tuple(data["shape"])
        )
