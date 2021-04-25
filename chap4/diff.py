from typing import List
import numpy as np
from collections.abc import Callable

from numpy.lib.function_base import diff


def gradient(f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
  h = 1e-4
  l: List[float] = []
  for i in range(len(x)):
    tmp = x[i]
    x[i] = tmp + h
    f1 = f(x)
    x[i] = tmp - h
    f2 = f(x)
    x[i] = tmp
    l.append((f1 - f2) / (2 * h))
  
  return np.array(l)


def f(x: np.ndarray) -> float:
  return np.sum(x * x)

x = np.array([3.0, 0])
print(gradient(f, x))
