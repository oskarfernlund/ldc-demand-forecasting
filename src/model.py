""" 
Bayesian linear regression model.
"""

from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.optimizers import Scipy

from src.basis import LinearBasis
from src.scalers import MinMaxScaler
from src.utils import ArrayOrTensor


class BLR(tf.Module):

    def __init__(self, data: Tuple[ArrayOrTensor, ArrayOrTensor]) -> None:
        
        X, y = data

        self.input_scaler = MinMaxScaler(X)
        self.output_scaler = MinMaxScaler(y)

        self.data = (
            tf.convert_to_tensor(self.input_scaler.scale(X), dtype=tf.float64),
            tf.convert_to_tensor(self.output_scaler.scale(y), dtype=tf.float64)
        )

        self.basis = LinearBasis()

        self.likelihood_variance = tfp.util.TransformedVariable(
            initial_value=0.1,
            bijector=tfp.bijectors.Softplus(),
            dtype=tf.float64,
            name="likelihood_variance"
        )

        self.prior_mean = tf.zeros((X.shape[1] + 1, 1), dtype=tf.float64)
        self.prior_cov = tf.eye(X.shape[1] + 1, dtype=tf.float64)

    @property
    def posterior_params(self):
        X, y = self.data
        Phi = self.basis(X)
        
        alpha = tf.math.scalar_mul(tf.math.reciprocal(self.likelihood_variance), Phi)
        beta = tf.linalg.matmul(alpha, Phi, transpose_a=True)
        gamma = tf.linalg.matmul(alpha, y, transpose_a=True)

        posterior_cov = tf.linalg.inv(beta + self.prior_cov)
        posterior_mean = tf.linalg.matmul(posterior_cov, gamma + self.prior_mean)
        
        return posterior_mean, posterior_cov

    def log_marginal_likelihood(self) -> tf.Tensor:
        X, y = self.data
        N = X.shape[0]
        Phi = self.basis(X)
        K = tf.linalg.matmul(Phi, Phi, transpose_b=True)
        I = tf.eye(N, dtype=tf.float64)
        L = tf.linalg.cholesky(K + tf.math.scalar_mul(self.likelihood_variance, I))
        alpha = tf.linalg.triangular_solve(L, y, lower=True)
        
        mahalanobis = -0.5 * tf.reduce_sum(tf.square(alpha))
        logdet = -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        const = -0.5 * N * np.log(2 * np.pi)

        return mahalanobis + logdet + const

    def predict(self, X: ArrayOrTensor) -> Tuple[np.ndarray, np.ndarray]:
        X = self.input_scaler.scale(X)
        N = X.shape[0]
        Phi = self.basis(X)
        posterior_mean, posterior_cov = self.posterior_params
        predictive_mean = tf.linalg.matmul(Phi, posterior_mean)
        I = tf.eye(N, dtype=tf.float64)
        noise = tf.math.scalar_mul(self.likelihood_variance, I)

        alpha = tf.linalg.matmul(posterior_cov, Phi, transpose_b=True)
        predictive_cov = tf.linalg.matmul(Phi, alpha) + noise

        predictive_var = tf.expand_dims(tf.linalg.tensor_diag_part(predictive_cov), axis=1)
        predictive_std = tf.math.sqrt(predictive_var)

        return (
            self.output_scaler.reverse_scale(predictive_mean).numpy(), 
            tf.square(self.output_scaler.reverse_scale(predictive_std)).numpy()
        )

    def train(self) -> None:
        
        def loss_closure():
            return -self.log_marginal_likelihood()

        optimizer = Scipy()
        optimizer.minimize(
            loss_closure,
            self.trainable_variables,
            method="L-BFGS-B",
            options={"maxiter": 100000},
        )