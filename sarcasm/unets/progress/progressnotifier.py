import math
import time
from tqdm import tqdm as tqdm

"""
Build a super dumb iterator wrapper which then returns a progress, this should be kinda splitted
first we need an object which remains alive the whole time where the logic and the variables are set to know
where the progress should be displayed (in our case we set in the gui, that we want the progress bar to be updated)

The second part is the iterator which only needs to be alive during "work"

thats how an iterator is made: https://treyhunner.com/2018/06/how-to-make-an-iterator-in-python/



this should probably be in an extra package since tqdm is used in the unet_pytorch package and we should not create
circular dependencies between the packages


the persistent class could have a private class which is the iterator and a method for creating the iterator


"""


class ProgressNotifier:

    def __init__(self):
        self.__task_progress = None
        self.__task_progress_details = None
        self.__use_tqdm = False

    @staticmethod
    def progress_notifier_tqdm():
        # should create a progress-notifier object prepared for using tqdm as output
        notifier = ProgressNotifier()
        notifier.__use_tqdm = True
        return notifier

    @staticmethod
    def silent_notifier():
        """
        returns a progress notifier that is silent (no progress notifications)
        used for the case a method requires a progress notifier but we do not want to show anything
        """
        return ProgressNotifier()

    def iterator(self, iterateable):
        # check if it is iterable
        try:
            iterator = iter(iterateable)
            if self.__use_tqdm:
                return tqdm(iterateable)
            return self.__IteratorWrapper(iterateable, self.__task_progress, self.__task_progress_details)
        except TypeError:
            raise TypeError("object is not possible to iterate")

    def set_progress_report(self, task):
        # check if task meets the requirements
        try:
            task(0)
            self.__task_progress = task
        except:
            raise Exception('The given variable is not a function with 1 numeric parameter (or similar constructs)')
        # sets the task that should be called on progress change
        # task should be a method or lambda that takes a number as input

    def set_progress_detail(self, task):
        # check if task meets the requirements
        try:
            task(0, 0, 0, 0, 0, 0)
            self.__task_progress_details = task
        except:
            raise Exception('The given variable is not a function with 6 numeric parameters (or similar constructs)')

        # sets the task that should be called on progress change
        # it gives details about the progress (like tqdm does), ETA, iterations per second and so on
        # this var doesn't have to be set

    class __IteratorWrapper:
        # this is a class variable (in java like static)
        timeMultiplier = 1000  # time values in milli seconds

        def __init__(self, iterable, task_progress, task_progress_details=None):
            self.__total = None
            self.__time = None  # time used until "now", now is the current "next" call
            self.__eta = None  # calculated value of how long will it take

            self.__iterable = iterable
            self.__iterator = iter(iterable)
            self.__task_progress = task_progress
            self.__task_progress_details = task_progress_details
            self.__time_start = int(round(time.time() * self.timeMultiplier))
            self.__current = 0  # current count of iterations
            # init total iterations number if possible
            if iterable is not None:
                try:
                    self.__total = len(iterable)
                except (TypeError, AttributeError):
                    self.__total = None
            if task_progress_details is not None:
                task_progress_details(0, 0, 0, 0, 0, 0)

        def __iter__(self):
            return self

        def __next__(self):
            self.__current += 1
            self.__time = int(round(time.time() * self.timeMultiplier))
            # eta is an approximation time until now divided by current and multiplied by total,
            # only if total is not None
            if self.__total is not None:
                self.__eta = ((self.__time - self.__time_start) / self.__current) * self.__total

            # report progress and return next element
            if self.__task_progress is not None:
                if self.__total is not None:  # if total is not None the current progress is reported
                    self.__task_progress(self.__current / self.__total)
                else:  # otherwise the number of iterations is reported
                    self.__task_progress(self.__current)

            # todo: the progress_details should update in an extra thread each second and the duration should be a timer in there
            # since there could be very slow iterations taking a while and with extra thread and duration as timer
            # the user "notices" progress
            if self.__task_progress_details is not None and self.__eta is not None:
                hh_eta = math.floor(self.__eta / (self.timeMultiplier * 3600))
                mm_eta = math.floor(self.__eta / (self.timeMultiplier * 60) - hh_eta * 60)
                ss_eta = math.floor((self.__eta / self.timeMultiplier) - (hh_eta * 3600 + mm_eta * 60))

                duration = (self.__time - self.__time_start) / self.timeMultiplier
                hh_current = math.floor(duration / 3600)
                mm_current = math.floor(duration / 60) - hh_current * 60
                ss_current = math.floor(duration) - (hh_current * 3600 + mm_current * 60)

                self.__task_progress_details(hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta)

            return self.__iterator.__next__()
