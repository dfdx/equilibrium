from functools import partial

import math
import jax
import jax.numpy as jnp
from flax import nnx
from diffusion.normalization import GroupNorm


class SinusoidalEmbedding(nnx.Module):

    def __init__(self, dim: int = 32):
        self.dim = dim

    def __call__(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        return emb


class TimeEmbedding(nnx.Module):

    def __init__(self, dim: int = 32, rngs: nnx.Rngs = nnx.Rngs(params=0)):
        self.dim = dim
        self.sin_embedding = SinusoidalEmbedding(dim)
        time_dim = self.dim * 4
        self.w1 = nnx.Linear(dim, time_dim, rngs=rngs)
        self.w2 = nnx.Linear(time_dim, time_dim, rngs=rngs)

    def __call__(self, t: jax.Array):
        se = self.sin_embedding(t)
        # Projecting the embedding into a 128 dimensional space
        t = self.w1(se)
        t = nnx.gelu(t)
        t = self.w2(t)
        return t



# Standard dot-product attention with eight heads.
class Attention(nnx.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            use_bias: bool = False,
            rngs: nnx.Rngs = nnx.Rngs(params=0)
    ):
        self.dim = dim
        self.num_heads = num_heads
        linear = partial(
            nnx.Linear, use_bias=use_bias, kernel_init=nnx.initializers.xavier_uniform(), rngs=rngs
        )
        self.wqkv = linear(dim, dim * 3)
        self.wo = linear(dim, dim)

    def __call__(self, x):
        bs, h, w, channels = x.shape
        x = x.reshape(bs, h*w, channels)
        bs, n, channels = x.shape
        qkv = self.wqkv(x)
        qkv = jnp.reshape(
            qkv, (bs, n, 3, self.num_heads, channels // self.num_heads)
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))   # [3, bs, n_heads, n=seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = (self.dim // self.num_heads) ** -0.5
        attention = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attention = nnx.softmax(attention, axis=-1)
        x = (attention @ v).swapaxes(1, 2).reshape(bs, n, channels)
        x = self.wo(x)
        x = jnp.reshape(x, (bs, int(x.shape[1]** 0.5), int(x.shape[1]** 0.5), -1))
        return x


# rename to ConvBlock?
class ConvBlock(nnx.Module):

    def __init__(self, dim: int = 32, groups: int = 8, rngs: nnx.Rngs = nnx.Rngs(params=0)):
        self.dim = dim
        self.groups = groups
        self.conv = nnx.Conv(self.dim, self.dim, (3, 3), rngs=rngs)
        self.norm = GroupNorm(num_features=dim, num_groups=self.groups, rngs=rngs)  # num_features=dim -- guessed

    def __call__(self, x):
        conv = self.conv(x)
        norm = self.norm(conv)
        activation = nnx.silu(norm)
        return activation


class ResnetBlock(nnx.Module):

    def __init__(self, dim: int = 32, groups: int = 8, rngs: nnx.Rngs = nnx.Rngs(params=0)):
        self.dim = dim
        self.groups = groups
        self.block1 = ConvBlock(dim, groups, rngs=rngs)
        self.dense = nnx.Linear(128, dim, rngs=rngs)   # 128 = original dim * 4 - size of time embeddings
        self.block2 = ConvBlock(dim, groups, rngs=rngs)
        self.conv = nnx.Conv(dim, dim, (1, 1), padding="SAME", rngs=rngs)   # 1st arg is guessed

    def __call__(self, x, t_emb=None):
        x = self.block1(x)
        if t_emb is not None:
            assert len(x.shape) == len(t_emb.shape) + 2, "Input and time embedding have incompatible shapes"
            t_emb = nnx.silu(t_emb)
            t_emb = self.dense(t_emb)
            x = jnp.expand_dims(jnp.expand_dims(t_emb, 1), 1) + x
        x = self.block2(x)
        res_conv = self.conv(x)
        return x + res_conv



class Down(nnx.Module):

    def __init__(self, in_channels: int, out_channels: int | None, num_groups: int, /, rngs: nnx.Rngs):
        self.resnet_block = ResnetBlock(in_channels, num_groups, rngs=rngs)
        self.attention = Attention(in_channels, rngs=rngs)
        self.norm = GroupNorm(num_features=in_channels, num_groups=num_groups, rngs=rngs)
        if out_channels:
            self.down_conv = nnx.Conv(in_channels, out_channels, (4,4), (2,2), rngs=rngs)

    def __call__(self, x, time_emb):
        x = self.resnet_block(x, time_emb)
        att = self.attention(x)
        norm = self.norm(att)
        x = norm + x
        if hasattr(self, "down_conv"):
            downsized = self.down_conv(x)
            return x, downsized
        else:
            return x, x


class Up(nnx.Module):

    def __init__(self, in_channels: int, out_channels: int, num_groups: int, /, rngs: nnx.Rngs):

        self.resnet_block = ResnetBlock(in_channels, num_groups, rngs=rngs)
        self.attention = Attention(in_channels, rngs=rngs)
        self.norm = GroupNorm(num_features=in_channels, num_groups=num_groups, rngs=rngs)
        self.up_conv = nnx.ConvTranspose(in_channels, out_channels, (4,4), (2,2), rngs=rngs)

    def __call__(self, x, time_emb):
        x = self.resnet_block(x, time_emb)
        att = self.attention(x)
        norm = self.norm(att)
        x = norm + x
        x = self.up_conv(x)
        return x


class UNet(nnx.Module):

    def __init__(self, dim: int = 8, dim_scale_factor: tuple = (1, 2, 4, 8), num_groups: int = 8, in_channels: int = 1, rngs: nnx.Rngs = nnx.Rngs(params=0)):
        dims = [dim * i for i in dim_scale_factor]
        self.dims = dims
        # self.dim = dim
        self.dim_scale_factor = dim_scale_factor
        self.num_groups = num_groups
        self.time_embedding = TimeEmbedding(dim)
        self.in_conv = nnx.Conv(in_channels, dim, (7, 7), padding=((3,3), (3,3)), rngs=rngs)
        # down
        # self.down_blocks = [Down(dims[i], dims[i + 1], num_groups, rngs=rngs) for i in range(len(dims))]
        self.down_blocks = []
        for i, dim in enumerate(dims):
            if i < len(dims) - 1:
                block = Down(dims[i], dims[i + 1], num_groups, rngs=rngs)
            else:
                block = Down(dims[i], None, num_groups, rngs=rngs)
            self.down_blocks.append(block)
        # self.down_convs = [nnx.Conv(self.dims[i], self.dims[i + 1], (4,4), (2,2), rngs=rngs) for i in range(len(self.dims) - 1)]
        # bottleneck
        mid_dim = dims[-1]
        self.mid_resnet = ResnetBlock(mid_dim, self.num_groups, rngs=rngs)
        self.mid_attn = Attention(dim, rngs=rngs)
        self.mid_norm = GroupNorm(num_features=mid_dim, num_groups=num_groups, rngs=rngs)
        self.mid_resnet2 = ResnetBlock(mid_dim, self.num_groups, rngs=rngs)
        # up
        # self.up_blocks = [Down(d * 2, num_groups, rngs=rngs) for d in reversed(dims)]


    def __call__(self, x, time):
        channels = x.shape[-1]
        x = self.in_conv(x)
        time_emb = self.time_embedding(time)

        # dims = [self.dim * i for i in self.dim_scale_factor]
        pre_downsampling = []

        # Downsampling phase
        for index, dim in enumerate(self.dims):
            skip, x = self.down_blocks[index](x, time_emb)
            # Saving this output for residual connection with the upsampling layer
            pre_downsampling.append(skip)
            # if index != len(self.dims) - 1:
            #     x = self.down_convs[index](x)

        # Middle block
        x = self.mid_resnet(x, time_emb)
        att = self.mid_attn(x)
        norm = self.mid_norm(att)
        x = norm + x
        x = self.mid_resnet2(x, time_emb)

        # Upsampling phase
        for index, dim in enumerate(reversed(self.dims)):
            x = jnp.concatenate([pre_downsampling.pop(), x], -1)
            # x = ResnetBlock(dim, self.num_groups)(x, time_emb)
            # x = ResnetBlock(dim, self.num_groups)(x, time_emb)
            # att = Attention(dim)(x)
            # norm = nn.GroupNorm(self.num_groups)(att)
            # x = norm + x
            if index != len(dims) - 1:
                x = nn.ConvTranspose(dim, (4,4), (2,2))(x)

        # Final ResNet block and output convolutional layer
        x = ResnetBlock(dim, self.num_groups)(x, time_emb)
        x = nn.Conv(channels, (1,1), padding="SAME")(x)
        return x


###########################################################


import tensorflow as tf
import tensorflow_datasets as tfds
BATCH_SIZE = 8


def get_datasets():
  # Load the MNIST dataset
    train_ds = tfds.load('mnist', as_supervised=True, split="train")

    # Normalization helper
    def preprocess(x, y):
        return tf.image.resize(tf.cast(x, tf.float32) / 127.5 - 1, (32, 32))

    # Normalize to [-1, 1], shuffle and batch
    train_ds = train_ds.map(preprocess, tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Return numpy arrays instead of TF tensors while iterating
    return tfds.as_numpy(train_ds)


def main():
    dim = 32  # kind of channels in intermediate layers
    bs, h, w, ch = 8, 32, 32, 1
    rngs = nnx.Rngs(params=0)
    train_ds = get_datasets()
    x = next(iter(train_ds))[:bs, :, :, :]

    time_embedding = TimeEmbedding(dim=dim, rngs=rngs)
    timestamps = jax.random.randint(jax.random.key(0), (bs,), minval=0, maxval=100)
    # t_emb = time_embedding(timestamps)[3:4, :]  # 3:4 - random timestamp generated by my brain

    # x = nnx.Conv(ch, dim // 3 * 2, (7, 7), padding=((3,3), (3,3)), rngs=rngs)(x)
    # x = nnx.Conv(ch, dim, (7, 7), padding=((3,3), (3,3)), rngs=rngs)(x)
    attn = Attention(dim=dim, rngs=rngs)
    attn(x)

    block = ConvBlock(dim=dim, rngs=rngs)
    block(x)

    rblock = ResnetBlock(dim=dim, rngs=rngs)
    rblock(x, t_emb)

    dblock = Down(in_channels=dim, num_groups=8, rngs=rngs)
    dblock(x, t_emb)

    self = UNet(dim=dim)
    time = timestamps


    # TODO: implement Up block