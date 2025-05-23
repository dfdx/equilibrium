import json
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.graph import Static

from fabrique.models.common.cache import KVCache, concatenate_to_cache
from fabrique.models.common.embeddings import (
    apply_rotary_pos_emb,
    create_sinusoidal_positions,
)
from fabrique.models.common.norm import RMSNorm
from fabrique.models.common.utils import padding_to_attention_mask
from fabrique.utils import check_and_update_fields
from equilibrium.models.embeddings import timestep_embedding


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_hidden_size: int = 14336
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    use_cache: bool = True

    @staticmethod
    def from_file(config_file: str, **kwargs):
        """
        Load ModelArgs from a Hugginface config file.
        """
        with open(config_file) as fp:
            config = json.load(fp)
        args = ModelArgs(
            dim=config["hidden_size"],
            n_layers=config["num_hidden_layers"],
            n_heads=config["num_attention_heads"],
            n_kv_heads=config["num_key_value_heads"],
            vocab_size=config["vocab_size"],
            # multiple_of=
            # ffn_dim_multiplier=
            ffn_hidden_size=config["intermediate_size"],
            norm_eps=config["rms_norm_eps"],
            # max_batch_size=
            max_seq_len=config["max_position_embeddings"],
        )
        return check_and_update_fields(args, **kwargs)


class Attention(nnx.Module):
    """
    Multi-head attention module.

    Args:
        args (ModelArgs): Model configuration parameters.
        sincos (jax.Array): Precomputed frequency array.
        full_causal_mask (jax.Array): Causal mask of size (1, max_seq_len, max_seq_len).
        rngs (nnx.Rngs): Random number generators.
    """

    def __init__(
        self, args: ModelArgs, sincos: Static, full_causal_mask: Static, rngs: nnx.Rngs
    ):
        self.args = args
        self.sincos = sincos
        self.full_causal_mask = full_causal_mask
        self.n_heads = args.n_heads
        self.n_kv_heads = self.n_heads if args.n_kv_heads is None else args.n_kv_heads

        self.n_rep = self.n_heads // self.n_kv_heads
        assert (
            self.args.dim % self.n_heads == 0
        ), f"Embedding size ({self.args.dim}) is not a multiplier of number of heads ({self.n_heads})"
        self.head_dim = self.args.dim // self.n_heads

        dense = partial(
            nnx.Linear,
            use_bias=False,
            dtype=self.args.dtype,
            param_dtype=self.args.param_dtype,
            kernel_init=jax.nn.initializers.normal(0.02),  # 0.02 - initializer range,
            rngs=rngs,
        )
        self.wq = dense(args.dim, self.n_heads * self.head_dim)
        self.wk = dense(args.dim, self.n_kv_heads * self.head_dim)
        self.wv = dense(args.dim, self.n_kv_heads * self.head_dim)
        self.wo = dense(self.n_heads * self.head_dim, self.args.dim)
        # if use_cache == False, we still create the variable to keep the same structure
        # but set its length to zero
        cache_len = self.args.max_seq_len if self.args.use_cache else 0
        cache_shape = (args.max_batch_size, cache_len, self.n_kv_heads, self.head_dim)
        self.cache_k = KVCache(jnp.zeros(cache_shape, args.param_dtype))
        self.cache_v = KVCache(jnp.zeros(cache_shape, args.param_dtype))

    def __call__(
        self,
        x: jax.Array,
        padding_mask: jax.Array
    ):
        """
        Forward pass of the attention module.

        Args:
            x (jax.Array): Input array.
            start_pos (int): Starting position for computations. Items before this
                position are taken from cache.
            padding_mask (jax.Array): Padding mask of size (bsz, kv_len), dtype = bool.

        Returns:
            jax.Array: Output array after attention.
        """
        start_pos = 0
        bsz, seq_len, _ = x.shape
        q_len = seq_len
        kv_len = seq_len

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, self.sincos.value, start_pos)


        # mask = jax.lax.dynamic_slice(
        #     self.full_causal_mask.value, (0, start_pos, 0), (1, q_len, kv_len)
        # )
        # mask = jnp.broadcast_to(mask, (bsz, *mask.shape[1:]))
        # if padding_mask is not None:
        #     pad_attn_mask = padding_to_attention_mask(padding_mask, shape=mask.shape)
        #     mask = nnx.combine_masks(mask, pad_attn_mask).astype(bool)  # type: ignore

        # using only padding mask
        mask = padding_to_attention_mask(padding_mask) # , shape=mask.shape)

        output = jax.nn.dot_product_attention(xq, xk, xv, mask=mask[:, None, :, :])
        output = output.reshape(output.shape[:2] + (self.args.dim,))
        return self.wo(output)


class FeedForward(nnx.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(params=0),
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (Linear): Linear transformation for the first layer.
            w2 (Linear): Linear transformation for the second layer.
            w3 (Linear): Linear transformation for the third layer.

        """
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.dtype = dtype
        self.param_dtype = param_dtype
        # hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if self.ffn_dim_multiplier is not None:
            hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
        hidden_dim = self.multiple_of * (
            (hidden_dim + self.multiple_of - 1) // self.multiple_of
        )
        linear = partial(
            nnx.Linear,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.w1 = linear(dim, hidden_dim)
        self.w2 = linear(hidden_dim, dim)
        self.w3 = linear(dim, hidden_dim)

    def __call__(self, x):
        return self.w2(nnx.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nnx.Module):

    def __init__(
        self, args: ModelArgs, sincos: Static, full_causal_mask: Static, rngs: nnx.Rngs
    ):
        """
        Initialize a TransformerBlock.

        Args:
            args (ModelArgs): Model configuration parameters.
            rngs (nnx.Rngs): Random number generator.
        """
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, sincos, full_causal_mask, rngs=rngs)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_hidden_size,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            dtype=self.args.dtype,
            param_dtype=self.args.param_dtype,
            rngs=rngs,
        )
        self.attention_norm = RMSNorm(
            args.dim, eps=args.norm_eps, param_dtype=args.param_dtype
        )
        self.ffn_norm = RMSNorm(
            args.dim, eps=args.norm_eps, param_dtype=args.param_dtype
        )

    def __call__(self, x: jax.Array, padding_mask: jax.Array):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (jax.Array): Input array.
            start_pos (int): Starting position for attention caching.
            padding_mask (jax.Array): Padding mask of size (bsz, kv_len), dtype = bool.
        Returns:
            jax.Array: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), padding_mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nnx.Module):

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs = nnx.Rngs(params=0)):
        """
        Initialize a Transformer model.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            args (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (nn.Embed): Token embeddings.
            layers (list): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (Linear): Linear layer for final output.
            sincos (jax.Array): Precomputed cosine and sine frequencies.

        """
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # self.tok_embeddings = nnx.Embed(
        #     num_embeddings=args.vocab_size,
        #     features=args.dim,
        #     dtype=args.dtype,
        #     param_dtype=args.param_dtype,
        #     rngs=rngs,
        # )

        sincos = Static(
            create_sinusoidal_positions(args.max_seq_len, args.dim // args.n_heads)
        )
        full_causal_mask = Static(
            nnx.make_causal_mask(jnp.ones(args.max_seq_len, dtype="bool"), dtype="bool")
        )

        self.layers = [
            TransformerBlock(args, sincos, full_causal_mask, rngs=rngs)
            for _ in range(args.n_layers)
        ]

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nnx.Linear(args.dim, args.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, padding_mask: jax.Array, t: jax.Array | None = None):
        h = x
        if t is not None:
            t_emb = timestep_embedding(t, x.shape[-1])[:, None, :]
            h += t_emb
        for i, layer in enumerate(self.layers):
            h = layer(h, padding_mask)
        h = self.norm(h)
        return h