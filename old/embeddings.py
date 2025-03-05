import math
import jax.numpy as jnp
from flax import nnx

# from https://github.com/madaan/minimal-text-diffusion/
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D array of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    )
    args = timesteps[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


# class SinusoidalEmbedding(nnx.Module):

#     def __init__(self, dim: int = 32):
#         self.dim = dim

#     def __call__(self, t):
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = jnp.exp(jnp.arange(half_dim) * -emb)
#         emb = t[:, None] * emb[None, :]
#         emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
#         return emb


# class TimeEmbedding(nnx.Module):

#     def __init__(self, dim: int = 32, rngs: nnx.Rngs = nnx.Rngs(params=0)):
#         self.dim = dim
#         self.sin_embedding = SinusoidalEmbedding(dim)
#         time_dim = self.dim * 4
#         self.w1 = nnx.Linear(dim, time_dim, rngs=rngs)
#         self.w2 = nnx.Linear(time_dim, time_dim, rngs=rngs)

#     def __call__(self, t: jax.Array):
#         se = self.sin_embedding(t)
#         # Projecting the embedding into a 128 dimensional space
#         t = self.w1(se)
#         t = nnx.gelu(t)
#         t = self.w2(t)
#         return t