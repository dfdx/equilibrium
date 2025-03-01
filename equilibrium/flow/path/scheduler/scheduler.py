# based on: https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/scheduler.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Union

import jax.numpy as jnp
from jax import grad

@dataclass
class SchedulerOutput:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        alpha_t (jnp.ndarray): :math:`\alpha_t`, shape (...).
        sigma_t (jnp.ndarray): :math:`\sigma_t`, shape (...).
        d_alpha_t (jnp.ndarray): :math:`\frac{\partial}{\partial t}\alpha_t`, shape (...).
        d_sigma_t (jnp.ndarray): :math:`\frac{\partial}{\partial t}\sigma_t`, shape (...).

    """

    alpha_t: jnp.ndarray = field(metadata={"help": "alpha_t"})
    sigma_t: jnp.ndarray = field(metadata={"help": "sigma_t"})
    d_alpha_t: jnp.ndarray = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: jnp.ndarray = field(metadata={"help": "Derivative of sigma_t."})


class Scheduler(ABC):
    """Base Scheduler class."""

    @abstractmethod
    def __call__(self, t: jnp.ndarray) -> SchedulerOutput:
        r"""
        Args:
            t (jnp.ndarray): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def snr_inverse(self, snr: jnp.ndarray) -> jnp.ndarray:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (jnp.ndarray): The signal-to-noise, shape (...)

        Returns:
            jnp.ndarray: t, shape (...)
        """
        ...


class ConvexScheduler(Scheduler):
    @abstractmethod
    def __call__(self, t: jnp.ndarray) -> SchedulerOutput:
        """Scheduler for convex paths.

        Args:
            t (jnp.ndarray): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def kappa_inverse(self, kappa: jnp.ndarray) -> jnp.ndarray:
        """
        Computes :math:`t` from :math:`\kappa_t`.

        Args:
            kappa (jnp.ndarray): :math:`\kappa`, shape (...)

        Returns:
            jnp.ndarray: t, shape (...)
        """
        ...

    def snr_inverse(self, snr: jnp.ndarray) -> jnp.ndarray:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (jnp.ndarray): The signal-to-noise, shape (...)

        Returns:
            jnp.ndarray: t, shape (...)
        """
        kappa_t = snr / (1.0 + snr)

        return self.kappa_inverse(kappa=kappa_t)


class CondOTScheduler(ConvexScheduler):
    """CondOT Scheduler."""

    def __call__(self, t: jnp.ndarray) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=jnp.ones_like(t),
            d_sigma_t=-jnp.ones_like(t),
        )

    def kappa_inverse(self, kappa: jnp.ndarray) -> jnp.ndarray:
        return kappa


class PolynomialConvexScheduler(ConvexScheduler):
    """Polynomial Scheduler."""

    def __init__(self, n: Union[float, int]) -> None:
        assert isinstance(
            n, (float, int)
        ), f"`n` must be a float or int. Got {type(n)=}."
        assert n > 0, f"`n` must be positive. Got {n=}."

        self.n = n

    def __call__(self, t: jnp.ndarray) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t**self.n,
            sigma_t=1 - t**self.n,
            d_alpha_t=self.n * (t ** (self.n - 1)),
            d_sigma_t=-self.n * (t ** (self.n - 1)),
        )

    def kappa_inverse(self, kappa: jnp.ndarray) -> jnp.ndarray:
        return jnp.power(kappa, 1.0 / self.n)


class VPScheduler(Scheduler):
    """Variance Preserving Scheduler."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__()

    def __call__(self, t: jnp.ndarray) -> SchedulerOutput:
        b = self.beta_min
        B = self.beta_max
        T = 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b
        dT = -(1 - t) * (B - b) - b

        return SchedulerOutput(
            alpha_t=jnp.exp(-0.5 * T),
            sigma_t=jnp.sqrt(1 - jnp.exp(-T)),
            d_alpha_t=-0.5 * dT * jnp.exp(-0.5 * T),
            d_sigma_t=0.5 * dT * jnp.exp(-T) / jnp.sqrt(1 - jnp.exp(-T)),
        )

    def snr_inverse(self, snr: jnp.ndarray) -> jnp.ndarray:
        T = -jnp.log(snr**2 / (snr**2 + 1))
        b = self.beta_min
        B = self.beta_max
        t = 1 - ((-b + jnp.sqrt(b**2 + 2 * (B - b) * T)) / (B - b))
        return t


class LinearVPScheduler(Scheduler):
    """Linear Variance Preserving Scheduler."""

    def __call__(self, t: jnp.ndarray) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=jnp.sqrt(1 - t**2),
            d_alpha_t=jnp.ones_like(t),
            d_sigma_t=-t / jnp.sqrt(1 - t**2),
        )

    def snr_inverse(self, snr: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(snr**2 / (1 + snr**2))


class CosineScheduler(Scheduler):
    """Cosine Scheduler."""

    def __call__(self, t: jnp.ndarray) -> SchedulerOutput:
        pi = jnp.pi
        return SchedulerOutput(
            alpha_t=jnp.sin(pi / 2 * t),
            sigma_t=jnp.cos(pi / 2 * t),
            d_alpha_t=pi / 2 * jnp.cos(pi / 2 * t),
            d_sigma_t=-pi / 2 * jnp.sin(pi / 2 * t),
        )

    def snr_inverse(self, snr: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * jnp.arctan(snr) / jnp.pi