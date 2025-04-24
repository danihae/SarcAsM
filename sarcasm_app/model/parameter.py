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


import threading
from inspect import signature
from PyQt5.QtWidgets import QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox, QComboBox
from typing import Callable, Union


class Parameter:
    """
    Utility class which enables storing of a parameter necessary for the application.
    It also enables binding the parameter to an UI element.
    """

    def __init__(self, name=None, value=None):
        self.__name = name
        self.__value = value
        self.__lambda_get_value = None
        self.__lambda_set_value = None
        self.__get_value_parser = None  # function to parse value before returning it (for example the parse frames)
        pass

    @property
    def name(self):
        return self.__name

    def backup_internals(self):
        return self.__lambda_get_value, self.__lambda_set_value, self.__get_value_parser

    def rollback_internals(self, get_lambda, set_lambda, parser_lambda):
        self.__lambda_set_value = set_lambda
        self.__lambda_get_value = get_lambda
        self.__get_value_parser = parser_lambda

    def __value_changed(self, opt_1=None, opt_2=None):
        if self.__lambda_get_value is not None:
            self.__value = self.__lambda_get_value()
            pass
        pass

    def set_value(self, value):
        old_value = self.__value
        self.__value = value
        if self.__lambda_set_value is not None:
            sig = signature(self.__lambda_set_value)
            if len(sig.parameters) == 1:
                self.__lambda_set_value(value)
            elif len(sig.parameters) >= 2:
                self.__lambda_set_value(value, old_value)
            pass
        pass

    def set_value_parser(self, value_parser):
        self.__get_value_parser = value_parser

    def get_value(self):
        if self.__get_value_parser is not None:
            return self.__get_value_parser(self.__value)
        return self.__value

    def get_value_as_int(self) -> int:
        return self.get_value()

    def get_value_as_float(self) -> float:
        return self.get_value()

    def get_raw_value(self):
        return self.__value

    def disconnect(self):
        self.__lambda_set_value = None
        self.__get_value_parser = None
        self.__lambda_get_value = None

    def connect(self, ui_element: Union[QDoubleSpinBox, QSpinBox, QLineEdit, QCheckBox, QComboBox, Callable]):
        """
        Connects the parameter to the ui element for pushing updates and retrieving updates
        """
        if isinstance(ui_element, QDoubleSpinBox):
            self.__lambda_set_value = ui_element.setValue
            self.__lambda_get_value = ui_element.value
            ui_element.valueChanged.connect(self.__value_changed)

            pass
        elif isinstance(ui_element, QSpinBox):
            self.__lambda_set_value = ui_element.setValue
            self.__lambda_get_value = ui_element.value
            ui_element.valueChanged.connect(self.__value_changed)
            pass
        elif isinstance(ui_element, QLineEdit):
            self.__lambda_set_value = lambda v: ui_element.setText(str(v))
            self.__lambda_get_value = ui_element.text
            # ui_element.editingFinished.connect(self.__value_changed)  # is not updating on programmatically changing text field value
            ui_element.textChanged.connect(self.__value_changed)
            pass
        elif isinstance(ui_element, QCheckBox):
            self.__lambda_set_value = ui_element.setChecked
            self.__lambda_get_value = ui_element.isChecked
            ui_element.stateChanged.connect(self.__value_changed)
            pass
        elif isinstance(ui_element, QComboBox):
            self.__lambda_set_value = ui_element.setCurrentText
            self.__lambda_get_value = ui_element.currentText
            ui_element.currentTextChanged.connect(self.__value_changed)
            pass
        elif callable(ui_element):
            # this part is to provide a parameter with callback on change method
            # it only needs a set_value lambda which calls the "ui_element" with old value and new value
            self.__lambda_set_value = ui_element
            pass
        # elif isinstance(ui_element, QRadioButton):
        #    self.__lambda_set_value = ui_element.setChecked
        #    self.__lambda_get_value = ui_element.isChecked
        #    ui_element.----.connect(self.__value_changed)
        #    pass
        if self.__lambda_set_value is not None and callable(self.__lambda_set_value):
            self.__lambda_set_value(self.__value)
        pass

    pass
