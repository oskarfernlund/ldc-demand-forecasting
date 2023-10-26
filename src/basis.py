""" 
Bases for regression.
"""

from abc import ABC, abstractmethod
from typing import Optional

import tensorflow as tf

from src.utils import ArrayOrTensor


class Basis(ABC, tf.Module):

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)

    def __call__(self, X: ArrayOrTensor) -> ArrayOrTensor:
        return self._compute_basis(X)

    @abstractmethod
    def _compute_basis(self, X: ArrayOrTensor) -> ArrayOrTensor:
        ...


class LinearBasis(Basis):

    def __init__(self) -> None:
        super().__init__(name="LinearBasis")

    @staticmethod
    def _compute_basis(X: ArrayOrTensor) -> ArrayOrTensor:
        return tf.concat(
            (
                tf.ones((X.shape[0], 1), dtype=tf.float64),
                tf.convert_to_tensor(X, dtype=tf.float64),
            ), axis=1
        )
