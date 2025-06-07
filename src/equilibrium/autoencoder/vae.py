import json
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from fabrique.models.common.cache import KVCache, concatenate_to_cache
from fabrique.models.common.embeddings import (
    apply_rotary_pos_emb,
    create_sinusoidal_positions,
)
from fabrique.models.common.norm import RMSNorm
from fabrique.models.common.utils import padding_to_attention_mask
from fabrique.utils import check_and_update_fields
from flax import nnx
from flax.nnx.graph import Static

from equilibrium.models.transformer import ModelArgs, TransformerBlock

# tokens -> Embed -> Encoder -> Decoder -> Classifier -> out_tokens


class Encoder(nnx.Module):

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

        self.tok_embeddings = nnx.Embed(
            num_embeddings=args.vocab_size,
            features=args.dim,
            dtype=args.dtype,
            param_dtype=args.param_dtype,
            rngs=rngs,
        )

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

    def __call__(
        self, tokens: jax.Array, start_pos: int, padding_mask: jax.Array | None = None
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (jax.Array): Input token indices.
            start_pos (int): Starting position for attention caching.
            padding_mask (jax.Array | None): Padding mask of size (bsz, kv_len), dtype = bool.

        Returns:
            jax.Array: Output logits after applying the Transformer model.
        """
        h = self.tok_embeddings(tokens)
        for i, layer in enumerate(self.layers):
            h = layer(
                h,
                start_pos,
                padding_mask=padding_mask,
            )
        h = self.norm(h)
        output = self.output(h).astype("float32")
        return output


class Encoder(nnx.Module):
    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        self.linear = nnx.Linear(input_size, intermediate_size, rngs=rngs)
        self.linear_mean = nnx.Linear(intermediate_size, output_size, rngs=rngs)
        self.linear_std = nnx.Linear(intermediate_size, output_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        x = self.linear(x)
        x = jax.nn.relu(x)

        mean = self.linear_mean(x)
        std = jnp.exp(self.linear_std(x))

        key = self.rngs.noise()
        z = mean + std * jax.random.normal(key, mean.shape)
        return z, mean, std


class Decoder(nnx.Module):
    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear1 = nnx.Linear(input_size, intermediate_size, rngs=rngs)
        self.linear2 = nnx.Linear(intermediate_size, output_size, rngs=rngs)

    def __call__(self, z: jax.Array) -> jax.Array:
        z = self.linear1(z)
        z = jax.nn.relu(z)
        logits = self.linear2(z)
        return logits


class VAE(nnx.Module):
    def __init__(
        self,
        image_shape: tuple[int, int],
        hidden_size: int,
        latent_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.image_shape = image_shape
        self.latent_size = latent_size
        input_size = image_shape[0] * image_shape[1]
        self.encoder = Encoder(input_size, hidden_size, latent_size, rngs=rngs)
        self.decoder = Decoder(latent_size, hidden_size, input_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        x = jax.vmap(jax.numpy.ravel)(x)  # flatten
        z, mean, std = self.encoder(x)
        logits = self.decoder(z)
        logits = jnp.reshape(logits, (-1, *self.image_shape))
        return logits, mean, std
