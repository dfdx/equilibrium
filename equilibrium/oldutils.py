import os
from typing import Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp


def unsqueeze_to_match(
    source: jnp.ndarray, target: jnp.ndarray, how: str = "suffix"
) -> jnp.ndarray:
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


def expand_tensor_like(
    input_tensor: jnp.ndarray, expand_to: jnp.ndarray
) -> jnp.ndarray:
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


def visualize_path(path, x0: jax.Array, x1: jax.Array, fig_path: str | None = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ValueError(
            f"visualize_path() requires matplotlib, but it's not installed"
        )
    xs = [path.sample(x0, x1, jnp.array([t])).x_t for t in jnp.arange(0, 1.1, 0.1)]
    xs = jnp.vstack(xs)
    plt.scatter(xs[:, 0], xs[:, 1])
    if fig_path is not None:
        plt.savefig(fig_path)
    else:
        plt.show()


def save_model(model, ckpt_dir: str):
    _, rng_state, state = nnx.split(model, nnx.RngState, ...)
    checkpointer = ocp.StandardCheckpointer()
    ckpt_path = os.path.abspath(os.path.join(ckpt_dir))
    checkpointer.save(ckpt_path, state)
    checkpointer.wait_until_finished()


cpu_device = jax.devices("cpu")[0]


def set_cpu_sharding(x):
    return jax.ShapeDtypeStruct(
        x.shape, x.dtype, sharding=jax.sharding.SingleDeviceSharding(cpu_device)
    )


def load_model(f: Callable, ckpt_dir: str, to_cpu: bool = False):
    ckpt_dir = os.path.abspath(os.path.join(ckpt_dir))
    abstract_model = nnx.eval_shape(f)
    graphdef, rng_state, abstract_state = nnx.split(abstract_model, nnx.RngState, ...)
    if to_cpu:
        abstract_state = jax.tree.map(set_cpu_sharding, abstract_state)
    checkpointer = ocp.StandardCheckpointer()
    state_restored = checkpointer.restore(ckpt_dir, abstract_state)
    model = nnx.merge(graphdef, rng_state, state_restored)
    return model


def plot_samples(samples: jax.Array, path: str | None = None):
    """
    Arguments:
    ----------
    samples : jax.Array
        Input image data of shape (B, H, W, C)
    """
    length = samples.shape[0]
    n_rows = int(jnp.sqrt(length))
    n_cols = int(jnp.ceil(length / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols)
    for i in range(length):
        sample = samples[i, :, :, :]
        sample = sample - sample.min()
        sample = sample / sample.max()
        r, c = i // n_cols, i % n_cols
        # print(f"r = {r}, c = {c}")
        axs[r, c].imshow(sample)
    plt.tight_layout()
    if path:
        fig.savefig(path)
