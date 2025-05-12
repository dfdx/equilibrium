"""
Adapted from https://github.com/facebookresearch/flow_matching/blob/main/examples/image/models/unet.py
"""

import math
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from equilibrium.models.embeddings import timestep_embedding

###############################################################################
#                                  Utils                                      #
###############################################################################


class GroupNorm32(nnx.GroupNorm):
    def __call__(self, x):
        return super().__call__(x.astype(jnp.float32)).astype(x.dtype)


def zero_module(module: nnx.Module) -> nnx.Module:
    """
    Zero out all parameters of a module.
    This assumes you can extract and update the module's parameter pytree.
    """
    params = nnx.state(module)
    zeroed = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    nnx.update(module, zeroed)
    return module


def spatial_dims(array_ndim: int, n_spatial_dims: int):
    """
    Return indices of spatial dimensions.

    Example:

        x = jax.random.normal(rngs(), (8, 16, 16, 3))   # 2D image in NHWC format
        dims = 2   # number of spatial dimensions (H and W)
        spatial_dims(x.ndim, dims)   # ==> [1, 2]
    """
    return list(range(array_ndim - n_spatial_dims - 1, array_ndim - 1))


###############################################################################
#                                Embeddings                                   #
###############################################################################


class ConstantEmbedding(nnx.Module):
    def __init__(self, in_channels, out_channels, *, dtype=jnp.float32, rngs: nnx.Rngs):
        scale = in_channels**0.5
        initializer = nnx.initializers.uniform(2 * scale, dtype=dtype)
        data = initializer(rngs(), (1, out_channels)) - scale
        self.embedding_table = nnx.Param(data)

    def __call__(self, emb):
        return self.embedding_table.repeat(emb.shape[0], 1)


###############################################################################
#                                 Attention                                   #
###############################################################################


class QKVAttention(nnx.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        self.n_heads = n_heads

    def __call__(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x T x x (3 * H * C)] tensor of Qs, Ks, and Vs.
        :return: an [N x T x (H * C)] tensor after attention.
        """
        bs, length, n_channels = qkv.shape
        assert n_channels % (3 * self.n_heads) == 0
        ch = n_channels // (3 * self.n_heads)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        # (N, T, H*C) -> (N, H*C, T) -> (N*H, C, T)
        q, k, v = [
            jnp.moveaxis(u, -1, -2).reshape(bs * self.n_heads, ch, length)
            for u in [q, k, v]
        ]
        weight = jnp.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = nnx.softmax(weight.astype(jnp.float32), axis=-1).astype(weight.dtype)
        out = jnp.einsum("bts,bcs->bct", weight, v)
        out = out.reshape(bs, -1, length)  # (N*H, C, T) -> (N, H*C, T)
        out = jnp.moveaxis(out, -1, -2)  # (N, H*C, T) -> (N, T, H*C)
        return out


class AttentionPool2d(nnx.Module):

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        *,
        output_dim: int = None,
        rngs: nnx.Rngs,
    ):
        # note: the order of HW and C dimensions is swapped compared to PyTorch impl
        self.positional_embedding = nnx.Param(
            jax.random.normal(rngs(), (spacial_dim**2 + 1, embed_dim)) / embed_dim**0.5
        )
        # self.qkv_proj = conv_nd(in_embed_dim, 3 * embed_dim, 1)
        self.qkv_proj = nnx.Conv(
            in_features=embed_dim, out_features=3 * embed_dim, kernel_size=1, rngs=rngs
        )
        # self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.c_proj = nnx.Conv(
            in_features=embed_dim,
            out_features=output_dim or embed_dim,
            kernel_size=1,
            rngs=rngs,
        )
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def __call__(self, x):
        b, *_spatial, c = x.shape
        x = x.reshape(b, -1, c)  # N(HW)C
        x = jnp.concatenate([x.mean(axis=-2, keepdims=True), x], axis=-2)  # N(HW+1)C
        x = x + self.positional_embedding[None, :, :].astype(x.dtype)  # N(HW+1)C
        x = self.qkv_proj(x)  # N(HW+1)(3*C)
        x = self.attention(x)  # N(HW+1)C
        x = self.c_proj(x)  # N(HW+1)C
        return x[:, :, 0]  # N(HW+1)


class AttentionBlock(nnx.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        *,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        rngs: nnx.Rngs,
    ):
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = GroupNorm32(num_features=channels, num_groups=channels, rngs=rngs)
        self.qkv = nnx.Conv(channels, channels * 3, (1,), rngs=rngs)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nnx.Conv(channels, channels, (1,), rngs=rngs))

    def __call__(self, x):
        b, *spatial, c = x.shape
        x = x.reshape(b, -1, c)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, *spatial, c)


###############################################################################
#                            Timestep Utils                                   #
###############################################################################


class TimestepBlock(nnx.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def __call__(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nnx.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def __call__(self, x, emb):
        for layer in self.layers:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


###############################################################################
#                              Unet Core Modules                              #
###############################################################################


class Upsample(nnx.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self, channels, use_conv, dims=2, *, out_channels=None, rngs: nnx.Rngs
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nnx.Conv(
                in_features=self.channels,
                out_features=self.out_channels,
                kernel_size=(3,) * dims,
                padding=1,
                rngs=rngs,
            )

    def __call__(self, x):
        assert x.shape[-1] == self.channels
        if self.dims == 3:
            x = jax.image.resize(
                x,
                (x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2, x.shape[4]),
                method="nearest",
            )
        else:
            # e.g. for NHWC image, spatial dims will be [1, 2]
            spatial_dims = [x.ndim - d - 2 for d in range(self.dims)]
            s = x.shape
            new_shape = [2 * s[i] if i in spatial_dims else s[i] for i in range(len(s))]
            x = jax.image.resize(x, new_shape, method="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nnx.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self, channels, use_conv, dims=2, *, out_channels=None, rngs: nnx.Rngs
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        strides = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nnx.Conv(
                self.channels,
                self.out_channels,
                (3,) * dims,
                strides=strides,
                padding=1,
                rngs=rngs,
            )
        else:
            assert self.channels == self.out_channels
            self.op = partial(
                nnx.avg_pool, window_dim=(strides,) * dims, strides=strides
            )

    def __call__(self, x):
        assert x.shape[-1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        *,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        emb_off=False,
        deterministic=False,
        rngs: nnx.Rngs,
    ):
        self.dims = dims
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nnx.Sequential(
            GroupNorm32(num_features=channels, num_groups=channels, rngs=rngs),
            nnx.silu,
            nnx.Conv(channels, self.out_channels, (3,) * dims, padding=1, rngs=rngs),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = lambda x: x

        if emb_off:
            self.emb_layers = ConstantEmbedding(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            )
        else:
            self.emb_layers = nnx.Sequential(
                nnx.silu,
                nnx.Linear(
                    emb_channels,
                    (
                        2 * self.out_channels
                        if use_scale_shift_norm
                        else self.out_channels
                    ),
                    rngs=rngs,
                ),
            )

        self.out_layers = nnx.Sequential(
            GroupNorm32(self.out_channels, self.out_channels, rngs=rngs),
            nnx.silu,
            nnx.Dropout(dropout, rngs=rngs, deterministic=deterministic),
            zero_module(
                nnx.Conv(
                    self.out_channels,
                    self.out_channels,
                    (3,) * dims,
                    padding=1,
                    rngs=rngs,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = lambda x: x
        elif use_conv:
            self.skip_connection = nnx.Conv(
                channels, self.out_channels, (3,) * dims, padding=1, rngs=rngs
            )
        else:
            self.skip_connection = nnx.Conv(
                channels, self.out_channels, (1,) * dims, rngs=rngs
            )

    def __call__(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x ... x C ] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x ... x C] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers.layers[:-1], self.in_layers.layers[-1]
            h = x
            for layer in in_rest:
                h = layer(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).astype(h.dtype)
        emb_out = jnp.expand_dims(emb_out, spatial_dims(h.ndim, self.dims))
        if self.use_scale_shift_norm:
            raise Exception("use_scale_shift_norm=True is not yet supported")
            # out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            # scale, shift = torch.chunk(emb_out, 2, dim=1)
            # h = out_norm(h) * (1 + scale) + shift
            # h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


@dataclass(eq=False)
class UNetModel(nnx.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    in_channels: int
    model_channels: int = 128
    out_channels: int = 3
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int] = (1, 2, 2, 2)
    dropout: float = 0.0
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: Optional[int] = None
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    with_fourier_features: bool = False
    ignore_time: bool = False
    input_projection: bool = True
    rngs: nnx.Rngs = None

    def __post_init__(self):
        assert isinstance(
            self.rngs, nnx.Rngs
        ), "Parameter `rngs` must be provided and be instance of nnx.Rngs"

        if self.with_fourier_features:
            self.in_channels += 12

        if self.num_heads_upsample == -1:
            self.num_heads_upsample = self.num_heads

        self.time_embed_dim = self.model_channels * 4
        if self.ignore_time:
            self.time_embed = lambda x: jnp.zeros(
                (x.shape[0], self.time_embed_dim), dtype=x.dtype
            )
        else:
            self.time_embed = nnx.Sequential(
                nnx.Linear(self.model_channels, self.time_embed_dim, rngs=self.rngs),
                nnx.silu,
                nnx.Linear(self.time_embed_dim, self.time_embed_dim, rngs=self.rngs),
            )

        if self.num_classes is not None:
            raise Exception("Parameter `num_classes` is not supported yet")
            # self.label_emb = nnx.Embed(
            #     self.num_classes + 1, self.time_embed_dim, padding_idx=self.num_classes
            # )

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        if self.input_projection:
            self.input_blocks = [
                TimestepEmbedSequential(
                    nnx.Conv(
                        self.in_channels,
                        ch,
                        (3,) * self.dims,
                        padding=1,
                        rngs=self.rngs,
                    )
                )
            ]
        else:
            self.input_blocks = [TimestepEmbedSequential(lambda x: x)]
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        emb_off=self.ignore_time and self.num_classes is None,
                        rngs=self.rngs,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            # use_new_attention_order=self.use_new_attention_order,
                            rngs=self.rngs,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                            emb_off=self.ignore_time and self.num_classes is None,
                            rngs=self.rngs,
                        )
                        if self.resblock_updown
                        else Downsample(
                            ch,
                            self.conv_resample,
                            dims=self.dims,
                            out_channels=out_ch,
                            rngs=self.rngs,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                emb_off=self.ignore_time and self.num_classes is None,
                rngs=self.rngs,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=self.use_checkpoint,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                # use_new_attention_order=self.use_new_attention_order,
                rngs=self.rngs,
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                emb_off=self.ignore_time and self.num_classes is None,
                rngs=self.rngs,
            ),
        )
        self._feature_size += ch

        self.output_blocks = []
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        emb_off=self.ignore_time and self.num_classes is None,
                        rngs=self.rngs,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            # use_new_attention_order=self.use_new_attention_order,
                            rngs=self.rngs,
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                            emb_off=self.ignore_time and self.num_classes is None,
                            rngs=self.rngs,
                        )
                        if self.resblock_updown
                        else Upsample(
                            ch,
                            self.conv_resample,
                            dims=self.dims,
                            out_channels=out_ch,
                            rngs=self.rngs,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nnx.Sequential(
            GroupNorm32(num_features=ch, num_groups=ch, rngs=self.rngs),
            nnx.silu,
            zero_module(
                nnx.Conv(
                    input_ch,
                    self.out_channels,
                    (3,) * self.dims,
                    padding=1,
                    rngs=self.rngs,
                )
            ),
        )

    def __call__(self, x, timesteps, extra):
        """
        Apply the model to an input batch.
        :param x: an [N x ... x C] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x ... x C] Tensor of outputs.
        """
        assert x.ndim == 4, "x should be 4D array, but is instead {x.ndim}D"
        assert (
            timesteps.ndim == 1
        ), f"timesteps should be 1D array, but is instead {timesteps.ndim}D"

        if self.with_fourier_features:
            raise Exception("Parameter with_fourier_features is not supported yet")
            # z_f = base2_fourier_features(x, start=6, stop=8, step=1)
            # x = torch.cat([x, z_f], dim=1)

        hs = []
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels).astype(x)
        )

        if self.ignore_time:
            emb = emb * 0.0

        # if self.num_classes and "label" not in extra:
        #     # Hack to deal with ddp find_unused_parameters not working with activation checkpointing...
        #     # self.num_classes corresponds to the pad index of the embedding table
        #     extra["label"] = torch.full(
        #         (x.size(0),), self.num_classes, dtype=torch.long, device=x.device
        #     )

        if self.num_classes is not None and "label" in extra:
            y = extra["label"]
            assert (
                y.shape == x.shape[:1]
            ), f"Labels have shape {y.shape}, which does not match the batch dimension of the input {x.shape}"
            emb = emb + self.label_emb(y)

        h = x
        if "concat_conditioning" in extra:
            raise Exception(f"Param concat_conditioning is not yet supported")
            # h = torch.cat([x, extra["concat_conditioning"]], dim=1)

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = module(h, emb)
        h = h.astype(x.dtype)
        result = self.out(h)
        return result


def example():
    in_channels = 3
    rngs = nnx.Rngs(0)
    self = UNetModel(in_channels, rngs=rngs)
    x = jax.random.normal(rngs(), (5, 64, 64, in_channels))
    timesteps = jax.random.uniform(rngs(), 5)
    y = self(x, timesteps, {})
    y = nnx.jit(self.__call__)(x, timesteps, {})


# # Based on https://github.com/google-research/vdm/blob/main/model_vdm.py
# def base2_fourier_features(
#     inputs: torch.Tensor, start: int = 0, stop: int = 8, step: int = 1
# ) -> torch.Tensor:
#     freqs = torch.arange(start, stop, step, device=inputs.device, dtype=inputs.dtype)

#     # Create Base 2 Fourier features
#     w = 2.0**freqs * 2 * np.pi
#     w = torch.tile(w[None, :], (1, inputs.size(1)))

#     # Compute features
#     h = torch.repeat_interleave(inputs, len(freqs), dim=1)
#     h = w[:, :, None, None] * h
#     h = torch.cat([torch.sin(h), torch.cos(h)], dim=1)
#     return h
