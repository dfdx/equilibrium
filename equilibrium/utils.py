from typing import Optional

import jax
import jax.numpy as jnp
from jax import grad


def unsqueeze_to_match(source: jnp.ndarray, target: jnp.ndarray, how: str = "suffix") -> jnp.ndarray:
    """
    Unsqueeze the source tensor to match the dimensionality of the target tensor.

    Args:
        source (jnp.ndarray): The source tensor to be unsqueezed.
        target (jnp.ndarray): The target tensor to match the dimensionality of.
        how (str, optional): Whether to unsqueeze the source tensor at the beginning
            ("prefix") or end ("suffix"). Defaults to "suffix".

    Returns:
        jnp.ndarray: The unsqueezed source tensor.
    """
    assert (
        how == "prefix" or how == "suffix"
    ), f"{how} is not supported, only 'prefix' and 'suffix' are supported."

    dim_diff = target.ndim - source.ndim

    for _ in range(dim_diff):
        if how == "prefix":
            source = jnp.expand_dims(source, 0)
        elif how == "suffix":
            source = jnp.expand_dims(source, -1)

    return source


def expand_tensor_like(input_tensor: jnp.ndarray, expand_to: jnp.ndarray) -> jnp.ndarray:
    """`input_tensor` is a 1d vector of length equal to the batch size of `expand_to`,
    expand `input_tensor` to have the same shape as `expand_to` along all remaining dimensions.

    Args:
        input_tensor (jnp.ndarray): (batch_size,).
        expand_to (jnp.ndarray): (batch_size, ...).

    Returns:
        jnp.ndarray: (batch_size, ...).
    """
    assert input_tensor.ndim == 1, "Input tensor must be a 1d vector."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    dim_diff = expand_to.ndim - input_tensor.ndim

    t_expanded = jnp.reshape(input_tensor, (-1, *([1] * dim_diff)))

    return jnp.broadcast_to(t_expanded, expand_to.shape)


# TODO: this looks incorrect, check and rewrite when needed
# def gradient(
#     output: jnp.ndarray,
#     x: jnp.ndarray,
#     grad_outputs: Optional[jnp.ndarray] = None,
#     create_graph: bool = False,
# ) -> jnp.ndarray:
#     """
#     Compute the gradient of the inner product of output and grad_outputs w.r.t :math:`x`.

#     Args:
#         output (jnp.ndarray): [N, D] Output of the function.
#         x (jnp.ndarray): [N, d_1, d_2, ... ] input
#         grad_outputs (Optional[jnp.ndarray]): [N, D] Gradient of outputs, if `None`,
#             then will use a tensor of ones
#         create_graph (bool): If True, graph of the derivative will be constructed, allowing
#             to compute higher order derivative products. Defaults to False.
#     Returns:
#         jnp.ndarray: [N, d_1, d_2, ... ]. the gradient w.r.t x.
#     """

#     if grad_outputs is None:
#         grad_outputs = jnp.ones_like(output)
#     grad_fn = grad(lambda x: jnp.sum(output * grad_outputs))
#     return grad_fn(x)



def visualize_path(path, x0: jax.Array, x1: jax.Array, fig_path: str | None = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ValueError(f"visualize_path() requires matplotlib, but it's not installed")
    xs = [path.sample(x0, x1, jnp.array([t])).x_t for t in jnp.arange(0, 1.1, 0.1)]
    xs = jnp.vstack(xs)
    plt.scatter(xs[:, 0], xs[:, 1])
    if fig_path is not None:
        plt.savefig(fig_path)
    else:
        plt.show()