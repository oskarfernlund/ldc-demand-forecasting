""" 
Feature scalers.
"""

import numpy as np
import tensorflow as tf

from src.utils import ArrayOrTensor

class MinMaxScaler:

    def __init__(self, data: np.ndarray) -> None:
        if tf.is_tensor(data):
            data = data.numpy()
        assert data.ndim == 2
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)

    def scale(self, data: ArrayOrTensor) -> ArrayOrTensor:
        return (data - self.min) / (self.max - self.min)

    def reverse_scale(self, data: ArrayOrTensor) -> ArrayOrTensor:
        return data * (self.max - self.min) + self.min
    