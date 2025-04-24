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


from .parameter import Parameter
from sarcasm import IOUtils


class Parameters:
    """
    The Parameters class contains a dict of parameters and handles loading and storing of those.
    """

    def __init__(self):
        # create a structure for storing all parameters, dictionary would be the easiest
        # would be nice if the parameters get updated on changing ui stuff
        self.__parameters_dict: dict[str, Parameter] = dict()
        pass

    def set_parameter(self, name: str, value=None):
        self.__parameters_dict[name] = Parameter(name=name, value=value)

    def get_parameter(self, name: str) -> Parameter:
        return self.__parameters_dict[name]

    def disconnect_parameters(self):
        for parameter in self.__parameters_dict.values():
            parameter.disconnect()

    def load(self, file_path_json_file):
        # load the parameters and set the values of all parameters accordingly

        simple_dict = IOUtils.json_deserialize(file_path_json_file)
        for key, value in simple_dict.items():
            if self.__parameters_dict.keys().__contains__(key):
                self.__parameters_dict[key].set_value(value)

        pass

    def store(self, file_path_json_file):
        simple_dict = dict()

        for parameter in self.__parameters_dict.values():
            simple_dict[parameter.name] = parameter.get_raw_value()
            pass

        IOUtils.json_serialize(simple_dict, file_path_json_file)
        pass

    pass
