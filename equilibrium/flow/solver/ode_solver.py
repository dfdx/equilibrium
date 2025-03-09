# TODO: unused, remove in the next round of cleaning
# based on: https://raw.githubusercontent.com/facebookresearch/flow_matching/refs/heads/main/flow_matching/solver/ode_solver.py

from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.ode import odeint

# from flow_matching.utils import gradient, ModelWrapper


class ODESolver:
    """A class to solve ordinary differential equations (ODEs) using a specified velocity model.

    This class utilizes a velocity field model to solve ODEs over a given time grid using numerical ode solvers.

    Args:
        velocity_model (Callable): a velocity field model receiving :math:`(x,t)` and returning :math:`u_t(x)`
    """

    def __init__(self, velocity_model: Callable):
        self.velocity_model = velocity_model

    def sample(
        self,
        x_init: Array,
        step_size: Optional[float],
        # method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Array = jnp.array([0.0, 1.0]),
        return_intermediates: bool = False,
        # enable_grad: bool = False,
        **model_extras,
    ) -> Union[Array, Sequence[Array]]:
        r"""Solve the ODE with the velocity field.

        Example:

        .. code-block:: python

            class DummyModel:

                def __call__(self, x: jax.Array, t: jax.Array, **extras) -> jax.Array:
                    return jnp.ones_like(x) * 3.0 * t**2

            velocity_model = DummyModel()
            solver = ODESolver(velocity_model=velocity_model)
            x_init = jnp.array([0.0, 0.0])
            step_size = 0.001
            time_grid = jnp.array([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Array): initial conditions (e.g., source samples :math:`X_0 \sim p`). Shape: [batch_size, ...].
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Array): The process is solved in the interval [min(time_grid, max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to torch.tensor([0.0, 1.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Defaults to False.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Array, Sequence[Array]]: The last timestep when return_intermediates=False, otherwise all values specified in time_grid.
        """

        def ode_func(t, x):
            return self.velocity_model(x=x, t=t, **model_extras)

        ode_opts = {"step_size": step_size} if step_size is not None else {}


        sol = odeint(
            ode_func,
            x_init,
            time_grid,
            # method=method,
            options=ode_opts,
            atol=atol,
            rtol=rtol,
        )

        if return_intermediates:
            return sol
        else:
            return sol[-1]

    def compute_likelihood(
        self,
        x_1: Array,
        log_p0: Callable[[Array], Array],
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Array = jnp.array([1.0, 0.0]),
        return_intermediates: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tuple[Array, Array], Tuple[Sequence[Array], Array]]:
        r"""Solve for log likelihood given a target sample at :math:`t=0`.

        Works similarly to sample, but solves the ODE in reverse to compute the log-likelihood. The velocity model must be differentiable with respect to x.
        The function assumes log_p0 is the log probability of the source distribution at :math:`t=0`.

        Args:
            x_1 (Tensor): target sample (e.g., samples :math:`X_1 \sim p_1`).
            log_p0 (Callable[[Tensor], Tensor]): Log probability function of the source distribution.
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): If step_size is None then time discretization is set by the time grid. Must start at 1.0 and end at 0.0, otherwise the likelihood computation is not valid. Defaults to torch.tensor([1.0, 0.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Otherwise only return the final sample. Defaults to False.
            exact_divergence (bool): Whether to compute the exact divergence or use the Hutchinson estimator.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]: Samples at time_grid and log likelihood values of given x_1.
        """
        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        # Fix the random projection for the Hutchinson divergence estimator
        if not exact_divergence:
            z = (jnp.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0

        def ode_func(x, t):
            return self.velocity_model(x=x, t=t, **model_extras)

        def dynamics_func(t, states):
            xt = states[0]
            xt.requires_grad_()
            ut = ode_func(xt, t)

            if exact_divergence:
                # Compute exact divergence
                div = 0
                for i in range(ut.flatten(1).shape[1]):
                    div += gradient(ut[:, i], xt, create_graph=True)[:, i]
            else:
                # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                ut_dot_z = torch.einsum(
                    "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                )
                grad_ut_dot_z = gradient(ut_dot_z, xt)
                div = torch.einsum(
                    "ij,ij->i",
                    grad_ut_dot_z.flatten(start_dim=1),
                    z.flatten(start_dim=1),
                )

            return ut.detach(), div.detach()

        y_init = (x_1, jnp.zeros(x_1.shape[0]))
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        sol, log_det = odeint(
            dynamics_func,
            y_init,
            time_grid,
            method=method,
            options=ode_opts,
            atol=atol,
            rtol=rtol,
        )

        x_source = sol[-1]
        source_log_p = log_p0(x_source)

        if return_intermediates:
            return sol, source_log_p + log_det[-1]
        else:
            return sol[-1], source_log_p + log_det[-1]