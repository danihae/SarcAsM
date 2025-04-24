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


class ChainExecution:
    """
    This class handles chain processing, process' which have multiple steps and each step is necessary for the next one.
    """

    def __init__(self, parameter_currently_processing, message_display_function=None):
        """
        parameter_currently_processing: should be of type sarcasm_app.model.Parameter
        message_display_function: a function/lambda which takes a string
        """
        self.__current_step = 0
        self.__execution_list = []
        self.__current_worker = None
        self.__currently_processing = parameter_currently_processing
        self.__message_display_function = message_display_function
        self.__get_lambda, self.__set_lambda, self.__parser_lambda = self.__currently_processing.backup_internals()
        self.__currently_processing.disconnect()

        pass

    def add_step(self, step):
        """
        Step needs to be a function which returns a Worker (see application_control.py::run_async_new)

        """
        self.__execution_list.append(step)
        pass

    def __on_currently_processing_changed(self, new, old=None):
        if old is True and new is False and self.__current_worker is not None and self.__current_worker.succeeded is True:
            # execute next step
            self.execute()
            pass
        elif old is True and new is False and self.__current_worker is not None and self.__current_worker.succeeded is False:
            if self.__message_display_function is not None:
                self.__message_display_function('chain execution exited due to not successful partial-operation')
                self.__currently_processing.rollback_internals(get_lambda=self.__get_lambda,
                                                               set_lambda=self.__set_lambda,
                                                               parser_lambda=self.__parser_lambda)
                self.__currently_processing.set_value(False)
            pass
        pass

    def __finishing_step(self):
        if self.__current_step >= len(self.__execution_list):
            if self.__message_display_function is not None:
                self.__message_display_function('chain execution exited successfully')
            self.__currently_processing.rollback_internals(get_lambda=self.__get_lambda,
                                                           set_lambda=self.__set_lambda,
                                                           parser_lambda=self.__parser_lambda)
            self.__currently_processing.set_value(False)
        pass

    def execute(self):
        if len(self.__execution_list) == 0:  # if there is nothing to execute, exit
            return
        elif self.__current_step == 0:  # on the first step make some preparations for finalizing
            self.__set_lambda(True)
            self.add_step(self.__finishing_step)
        elif len(self.__execution_list) <= self.__current_step:  # execution finished
            return

        self.__currently_processing.disconnect()
        step = self.__current_step
        self.__current_step += 1  # storing the value and increasing before the next step, enables checking
        # the next step in the called method -> check if finished
        self.__current_worker = self.__execution_list[step]()
        if self.__current_step < len(self.__execution_list):
            self.__currently_processing.connect(self.__on_currently_processing_changed)

            pass

        pass

    pass
