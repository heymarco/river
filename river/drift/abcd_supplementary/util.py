import numpy as np


def handles_dicts(func):
    def handles_dicts_wrapper(self, x):
        if isinstance(x, dict):
            x = np.array(list(x.values()))
        return func(self, x)
    return handles_dicts_wrapper


def convert_to_univariate_if_possible(func):
    def convert_to_univariate_if_possible_wrapper(self, x):
        if isinstance(x, float):
            return func(self, x)
        if x.shape[-1] == 1:
            x = x[0]
        return func(self, x)
    return convert_to_univariate_if_possible_wrapper
