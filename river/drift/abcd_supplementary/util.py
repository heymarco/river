import numbers
from typing import Union

import numpy as np


def handles_dicts(func):
    def handles_dicts_wrapper(x: Union[numbers.Number, np.ndarray, dict], *args, **kwargs):
        if isinstance(x, dict):
            x = np.ndarray(x.values())
        return func(x, *args, **kwargs)
    return handles_dicts_wrapper
