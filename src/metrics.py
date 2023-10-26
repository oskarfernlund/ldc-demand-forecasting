""" 
Evaluation metrics.
"""

import numpy as np

from scipy.stats import norm


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(y_pred - y_true))).item()


def gaussian_nlpd(
    mu_pred: np.ndarray, std_pred: np.ndarray, y_true: np.ndarray
) -> float:
    return -np.mean(norm.logpdf(x=y_true, loc=mu_pred, scale=std_pred))
