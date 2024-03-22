import codecs
import copy
import json
import pickle
import numpy as np
from numpy import ndarray


class IOUtils:

    # region JSON Serialization --- works with top1.csv
    @staticmethod
    def __serialize_field(field):

        if isinstance(field, ndarray):
            return {'type': 'ndarray', 'values': np.array(field).tolist()}
        elif isinstance(field, list):
            list_new = []
            for val in field:
                list_new.append(IOUtils.__serialize_field(val))
            return list_new
        elif isinstance(field, dict):
            for key in field:
                field[key] = IOUtils.__serialize_field(field[key])
            return field
        elif isinstance(field, np.generic):
            return {'value': field.item(), 'type': field.dtype.name}
        else:
            return field

    @staticmethod
    def __deserialize_field(field):

        if isinstance(field, list):
            list_new = []
            for val in field:
                list_new.append(IOUtils.__deserialize_field(val))
            return list_new
        elif isinstance(field, dict) and 'type' in field and ('value' in field or 'values' in field):
            if 'values' in field:
                return np.array(field['values'])
            else:
                if field['type'] == 'int32':
                    return np.int32(field['value'])
                elif field['type'] == 'float32':
                    return np.float32(field['value'])
                elif field['type'] == 'float64':
                    return np.float64(field['value'])
        elif isinstance(field, dict):
            for key in field:
                field[key] = IOUtils.__deserialize_field(field[key])
            return field
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

    # endregion

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
