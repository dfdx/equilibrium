from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import random

from equilibrium.flow.path.path_sample import PathSample


class ProbPath(ABC):
    r"""Abstract class, representing a probability path.

    A probability path transforms the distribution :math:`p(X_0)` into :math:`p(X_1)` over :math:`t=0\rightarrow 1`.

    The ``ProbPath`` class is designed to support model training in the flow matching framework. It supports two key functionalities: (1) sampling the conditional probability path and (2) conversion between various training objectives.
    Here is a high-level example

    .. code-block:: python

        # Instantiate a probability path
        my_path = ProbPath(...)

        for x_0, x_1 in dataset:
            # Sets t to a random value in [0,1]
            key = random.PRNGKey(0)
            t = random.uniform(key)

            # Samples the conditional path X_t ~ p_t(X_t|X_0,X_1)
            path_sample = my_path.sample(x_0=x_0, x_1=x_1, t=t)

            # Optimizes the model. The loss function varies, depending on model and path.
            loss(path_sample, my_model(x_t, t)).backward()

    """

    @abstractmethod
    def sample(self, x_0: jnp.ndarray, x_1: jnp.ndarray, t: jnp.ndarray) -> PathSample:
        r"""Sample from an abstract probability path:

        | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)`.
        | returns :math:`X_0, X_1, X_t \sim p_t(X_t)`, and a conditional target :math:`Y`, all objects are under ``PathSample``.

        Args:
            x_0 (jnp.ndarray): source data point, shape (batch_size, ...).
            x_1 (jnp.ndarray): target data point, shape (batch_size, ...).
            t (jnp.ndarray): times in [0,1], shape (batch_size).

        Returns:
            PathSample: a conditional sample.
        """

    def assert_sample_shape(self, x_0: jnp.ndarray, x_1: jnp.ndarray, t: jnp.ndarray):
        assert (
            t.ndim == 1
        ), f"The time vector t must have shape [batch_size]. Got {t.shape}."
        assert (
            t.shape[0] == x_0.shape[0] == x_1.shape[0]
        ), f"Time t dimension must match the batch size [{x_1.shape[0]}]. Got {t.shape}"
