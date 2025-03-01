# based on: https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/affine.py
from equilibrium.flow.path.path import ProbPath
from equilibrium.flow.path.path_sample import PathSample
from equilibrium.flow.path.scheduler.scheduler import CondOTScheduler, Scheduler
from equilibrium.utils import expand_tensor_like
import jax.numpy as jnp


class AffineProbPath(ProbPath):
    r"""The ``AffineProbPath`` class represents a specific type of probability path where the transformation between distributions is affine.
    An affine transformation can be represented as:

    .. math::

        X_t = \alpha_t X_1 + \sigma_t X_0,

    where :math:`X_t` is the transformed data point at time `t`. :math:`X_0` and :math:`X_1` are the source and target data points, respectively. :math:`\alpha_t` and :math:`\sigma_t` are the parameters of the affine transformation at time `t`.

    The scheduler is responsible for providing the time-dependent parameters :math:`\alpha_t` and :math:`\sigma_t`, as well as their derivatives, which define the affine transformation at any given time `t`.

    Using ``AffineProbPath`` in the flow matching framework:

    .. code-block:: python

        # Instantiates a probability path
        my_path = AffineProbPath(...)
        mse_loss = jax.nn.MSELoss()   # TODO

        for x_1 in dataset:
            # Sets x_0 to random noise
            x_0 = jax.random.normal()

            # Sets t to a random value in [0,1]
            t = jax.random.uniform()

            # Samples the conditional path X_t ~ p_t(X_t|X_0,X_1)
            path_sample = my_path.sample(x_0=x_0, x_1=x_1, t=t)

            # Computes the MSE loss w.r.t. the velocity
            loss = mse_loss(path_sample.dx_t, my_model(x_t, t))
            loss.backward()

    Args:
        scheduler (Scheduler): An instance of a scheduler that provides the parameters :math:`\alpha_t`, :math:`\sigma_t`, and their derivatives over time.

    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def sample(self, x_0: jnp.ndarray, x_1: jnp.ndarray, t: jnp.ndarray) -> PathSample:
        r"""Sample from the affine probability path:

        | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)` and a scheduler :math:`(\alpha_t,\sigma_t)`.
        | return :math:`X_0, X_1, X_t = \alpha_t X_1 + \sigma_t X_0`, and the conditional velocity at :math:`X_t, \dot{X}_t = \dot{\alpha}_t X_1 + \dot{\sigma}_t X_0`.

        Args:
            x_0 (jnp.ndarray): source data point, shape (batch_size, ...).
            x_1 (jnp.ndarray): target data point, shape (batch_size, ...).
            t (jnp.ndarray): times in [0,1], shape (batch_size).

        Returns:
            PathSample: a conditional sample at :math:`X_t \sim p_t`.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        scheduler_output = self.scheduler(t)

        alpha_t = expand_tensor_like(
            input_tensor=scheduler_output.alpha_t, expand_to=x_1
        )
        sigma_t = expand_tensor_like(
            input_tensor=scheduler_output.sigma_t, expand_to=x_1
        )
        d_alpha_t = expand_tensor_like(
            input_tensor=scheduler_output.d_alpha_t, expand_to=x_1
        )
        d_sigma_t = expand_tensor_like(
            input_tensor=scheduler_output.d_sigma_t, expand_to=x_1
        )

        # construct xt ~ p_t(x|x1).
        x_t = sigma_t * x_0 + alpha_t * x_1
        dx_t = d_sigma_t * x_0 + d_alpha_t * x_1

        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)

    def target_to_velocity(self, x_1: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        r"""Convert from x_1 representation to velocity.

        | given :math:`X_1`.
        | return :math:`\dot{X}_t`.

        Args:
            x_1 (jnp.ndarray): target data point.
            x_t (jnp.ndarray): path sample at time t.
            t (jnp.ndarray): time in [0,1].

        Returns:
            jnp.ndarray: velocity.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = d_sigma_t / sigma_t
        b_t = (d_alpha_t * sigma_t - d_sigma_t * alpha_t) / sigma_t

        return a_t * x_t + b_t * x_1

    def epsilon_to_velocity(self, epsilon: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        r"""Convert from epsilon representation to velocity.

        | given :math:`\epsilon`.
        | return :math:`\dot{X}_t`.

        Args:
            epsilon (jnp.ndarray): noise in the path sample.
            x_t (jnp.ndarray): path sample at time t.
            t (jnp.ndarray): time in [0,1].

        Returns:
            jnp.ndarray: velocity.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = d_alpha_t / alpha_t
        b_t = (d_sigma_t * alpha_t - d_alpha_t * sigma_t) / alpha_t

        return a_t * x_t + b_t * epsilon

    def velocity_to_target(self, velocity: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        r"""Convert from velocity to x_1 representation.

        | given :math:`\dot{X}_t`.
        | return :math:`X_1`.

        Args:
            velocity (jnp.ndarray): velocity at the path sample.
            x_t (jnp.ndarray): path sample at time t.
            t (jnp.ndarray): time in [0,1].

        Returns:
            jnp.ndarray: target data point.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = -d_sigma_t / (d_alpha_t * sigma_t - d_sigma_t * alpha_t)
        b_t = sigma_t / (d_alpha_t * sigma_t - d_sigma_t * alpha_t)

        return a_t * x_t + b_t * velocity

    def epsilon_to_target(self, epsilon: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        r"""Convert from epsilon representation to x_1 representation.

        | given :math:`\epsilon`.
        | return :math:`X_1`.

        Args:
            epsilon (jnp.ndarray): noise in the path sample.
            x_t (jnp.ndarray): path sample at time t.
            t (jnp.ndarray): time in [0,1].

        Returns:
            jnp.ndarray: target data point.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        sigma_t = scheduler_output.sigma_t

        a_t = 1 / alpha_t
        b_t = -sigma_t / alpha_t

        return a_t * x_t + b_t * epsilon

    def velocity_to_epsilon(self, velocity: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        r"""Convert from velocity to noise representation.

        | given :math:`\dot{X}_t`.
        | return :math:`\epsilon`.

        Args:
            velocity (jnp.ndarray): velocity at the path sample.
            x_t (jnp.ndarray): path sample at time t.
            t (jnp.ndarray): time in [0,1].

        Returns:
            jnp.ndarray: noise in the path sample.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = -d_alpha_t / (d_sigma_t * alpha_t - d_alpha_t * sigma_t)
        b_t = alpha_t / (d_sigma_t * alpha_t - d_alpha_t * sigma_t)

        return a_t * x_t + b_t * velocity

    def target_to_epsilon(self, x_1: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        r"""Convert from x_1 representation to velocity.

        | given :math:`X_1`.
        | return :math:`\epsilon`.

        Args:
            x_1 (jnp.ndarray): target data point.
            x_t (jnp.ndarray): path sample at time t.
            t (jnp.ndarray): time in [0,1].

        Returns:
            jnp.ndarray: noise in the path sample.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        sigma_t = scheduler_output.sigma_t

        a_t = 1 / sigma_t
        b_t = -alpha_t / sigma_t

        return a_t * x_t + b_t * x_1


class CondOTProbPath(AffineProbPath):
    r"""The ``CondOTProbPath`` class represents a conditional optimal transport probability path.

    This class is a specialized version of the ``AffineProbPath`` that uses a conditional optimal transport scheduler to determine the parameters of the affine transformation.

    The parameters :math:`\alpha_t` and :math:`\sigma_t` for the conditional optimal transport path are defined as:

    .. math::

        \alpha_t = t \quad \text{and} \quad \sigma_t = 1 - t.
    """

    def __init__(self):
        self.scheduler = CondOTScheduler()