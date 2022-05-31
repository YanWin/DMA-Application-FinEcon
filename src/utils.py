import sys
from pathlib import Path
import numpy as np


def get_project_root() -> Path:
    return Path(__file__).parent.parent

def inverse(x):
    if isinstance(x, np.ndarray):
        if len(x) == 1:
            return 1/x
        elif len(x) > 1:
            return np.linalg.inv(x)
    elif isinstance(x, (int, float)):
        return 1/x
    else:
        print("Value to invert" + str(x))
        raise Exception("Cant calculate inverse")

def diag(x):
    if isinstance(x, np.ndarray):
        if len(x) == 1:
            return x
        elif len(x) > 1:
            return np.diag(x)
    elif isinstance(x, (int, float)):
        return x
    else:
        print("Take diagonal from" + str(x))
        raise Exception("Cant take diagonal elements")


