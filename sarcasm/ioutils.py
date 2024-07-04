# Usage of this software for commercial purposes without a license is strictly prohibited.

import codecs
import copy
import json
import pickle

import numpy as np
from scipy import sparse


class IOUtils:
    """Utility functions for storing and loading IO data"""
    # region JSON Serialization --- works with top1.csv
    @staticmethod
    def __serialize_field(field):
        if sparse.issparse(field):  # Check if the field is a sparse matrix
            return {'type': 'sparse_matrix', 'values': IOUtils.__sparse_to_json_serializable(field)}
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
        json.dump(cpy, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    @staticmethod
    def json_deserialize(file_path):
        obj = json.load(codecs.open(file_path, 'r', encoding='utf-8'))
        obj = IOUtils.__deserialize_field(obj)
        return obj

    @staticmethod
    def serialize_profile_data(start, end, profiles, line_width, file_path):
        result_obj = {
            "line_start_x": start[1],
            "line_start_y": start[0],
            "line_end_x": end[1],
            "line_end_y": end[0],
            "line_width": line_width,
            "profiles_description": "array[image in stack - time slot][profile_values]",  # just for documentation
            "profiles": profiles  # tolist is used because numpy arrays are not serializable by json
        }
        IOUtils.json_serialize(result_obj, file_path)

    @staticmethod
    def serialize_pickle(obj, file_path):
        pickle.dump(obj, open(file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialize_pickle(file_path):
        return pickle.load(open(file_path, 'rb'))

    @staticmethod
    def serialize_profile_data_pickle(start, end, profiles, line_width, file_path):
        result_obj = {
            "line_start_x": start[1],
            "line_start_y": start[0],
            "line_end_x": end[1],
            "line_end_y": end[0],
            "line_width": line_width,
            "profiles_description": "array[image in stack - time slot][profile_values]",  # just for documentation
            "profiles": profiles  # tolist is used because numpy arrays are not serializable by json
        }
        IOUtils.serialize_pickle(result_obj, file_path)

    @staticmethod
    def __sparse_to_json_serializable(sparse_matrix):
        """
        Converts a sparse matrix to a JSON-serializable dictionary.
        """
        # Convert to COO format
        sparse_coo = sparse_matrix.tocoo()

        # Prepare a serializable object
        serializable_data = {
            "data": sparse_coo.data.tolist(),
            "row": sparse_coo.row.tolist(),
            "col": sparse_coo.col.tolist(),
            "shape": sparse_coo.shape
        }

        return json.dumps(serializable_data)

    @staticmethod
    def __json_serializable_to_sparse(json_data):
        """
        Converts a JSON-serializable dictionary back to a COO sparse matrix.
        """
        data = json.loads(json_data)

        # Extract data for COO sparse matrix creation
        coo_data = np.array(data["data"])
        row = np.array(data["row"])
        col = np.array(data["col"])
        shape = tuple(data["shape"])

        # Create and return the COO sparse matrix
        return sparse.coo_matrix((coo_data, (row, col)), shape=shape)
