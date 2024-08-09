# from PR: https://github.com/google/flax/pull/4095/files

import typing as tp

import jax
import jax.numpy as jnp

from flax import nnx
from flax.nnx.nnx import rnglib
from flax.nnx.nnx.module import Module
from flax.nnx.nnx.nn import  initializers
from flax.typing import (
  Dtype,
  Initializer,
  Axes,
)

from flax.nnx.nnx.nn.normalization import (
  _normalize,
  _compute_stats,
  _canonicalize_axes
)



class GroupNorm(Module):
  """Group normalization (arxiv.org/abs/1803.08494).

  This op is similar to batch normalization, but statistics are shared across
  equally-sized groups of channels and not shared across batch dimension.
  Thus, group normalization does not depend on the batch composition and does
  not require maintaining internal state for storing statistics.
  The user should either specify the total number of channel groups or the
  number of channels per group.

  .. note::
    LayerNorm is a special case of GroupNorm where ``num_groups=1``.

  Example usage::

    >>> from flax import nnx
    >>> import jax
    >>> import numpy as np
    ...
    >>> x = jax.random.normal(jax.random.key(0), (3, 4, 5, 6))
    >>> layer = nnx.GroupNorm(num_features=6, num_groups=3, rngs=nnx.Rngs(0))
    >>> nnx.state(layer)
    State({
      'bias': VariableState(
        type=Param,
        value=Array([0., 0., 0., 0., 0., 0.], dtype=float32)
      ),
      'scale': VariableState(
        type=Param,
        value=Array([1., 1., 1., 1., 1., 1.], dtype=float32)
      )
    })
    >>> y = layer(x)
    ...
    >>> y = nnx.GroupNorm(num_features=6, num_groups=1, rngs=nnx.Rngs(0))(x)
    >>> y2 = nnx.LayerNorm(num_features=6, reduction_axes=(1, 2, 3), rngs=nnx.Rngs(0))(x)
    >>> np.testing.assert_allclose(y, y2)

  Attributes:
    num_features: the number of input features/channels.
    num_groups: the total number of channel groups. The default value of 32 is
      proposed by the original group normalization paper.
    group_size: the number of channels in a group.
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    reduction_axes: List of axes used for computing normalization statistics.
      This list must include the final dimension, which is assumed to be the
      feature axis. Furthermore, if the input used at call time has additional
      leading axes compared to the data used for initialisation, for example due
      to batching, then the reduction axes need to be defined explicitly.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See ``jax.pmap`` for a description of axis names (default: None).
      This is only needed if the model is subdivided across devices, i.e. the
      array being normalized is sharded across devices within a pmap or shard
      map. For SPMD jit, you do not need to manually synchronize. Just make sure
      that the axes are correctly annotated and XLA:SPMD will insert the
      necessary collectives.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over the
      examples on the first two and last two devices. See ``jax.lax.psum`` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
    rngs: rng key.
  """

  def __init__(
    self,
    num_features: int,
    num_groups: tp.Optional[int] = 32,
    group_size: tp.Optional[int] = None,
    *,
    epsilon: float = 1e-6,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    use_bias: bool = True,
    use_scale: bool = True,
    bias_init: Initializer = initializers.zeros_init(),
    scale_init: Initializer = initializers.ones_init(),
    reduction_axes: tp.Optional[Axes] = None,
    axis_name: tp.Optional[str] = None,
    axis_index_groups: tp.Any = None,
    use_fast_variance: bool = True,
    rngs: rnglib.Rngs,
  ):
    self.feature_axis = -1

    if (num_groups is None and group_size is None) or (
      num_groups is not None and group_size is not None
    ):
      raise ValueError(
        'Either `num_groups` or `group_size` should be '
        'specified. If `group_size` is to be specified, '
        'pass `num_groups=None` as argument to override '
        'the default `num_groups` value of 32.'
      )

    if group_size is not None:
      if num_features % group_size != 0:
        raise ValueError(
          'Number of features ({}) is not multiple of the '
          'group size ({}).'.format(num_features, group_size)
        )
      self.num_groups = num_features // group_size
      self.group_size = group_size
    else:
      if not isinstance(num_groups, int) or num_groups <= 0 or (
        num_features % num_groups != 0
      ):
        raise ValueError(
          'Number of groups ({}) does not divide the number'
          ' of channels ({}).'.format(num_groups, num_features)
        )
      self.num_groups = num_groups
      self.group_size = num_features // num_groups

    feature_shape = (num_features,)
    if use_scale:
      key = rngs.params()
      self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
    else:
      self.scale = nnx.Param(None)

    if use_bias:
      key = rngs.params()
      self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
    else:
      self.bias = nnx.Param(None)

    self.epsilon = epsilon
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.use_bias = use_bias
    self.use_scale = use_scale
    self.bias_init = bias_init
    self.scale_init = scale_init
    self.reduction_axes = reduction_axes
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups
    self.use_fast_variance = use_fast_variance

  def __call__(self, x, *, mask: tp.Optional[jax.Array] = None):
    """Applies group normalization to the input (arxiv.org/abs/1803.08494).

    Args:
      x: the input of shape ``...self.num_features`` where ``self.num_features``
        is a channels dimension and ``...`` represents an arbitrary number of
        extra dimensions that can be used to accumulate statistics over. If no
        reduction axes have been specified then all additional dimensions ``...``
        will be used to accumulate statistics apart from the leading dimension
        which is assumed to represent the batch.
      mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
        the positions for which the mean and variance should be computed.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    if self.reduction_axes is not None:
      reduction_axes = self.reduction_axes
    else:
      reduction_axes = list(range(1, x.ndim - 1)) + [-1]
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)

    group_shape = x.shape[:-1] + (self.num_groups, self.group_size)
    if mask is not None:
      mask = mask.reshape(mask.shape[:-1] + (self.num_groups, self.group_size))

    mean, var = _compute_stats(
      x.reshape(group_shape),
      list(reduction_axes[:-1]) + [-1],
      self.dtype,
      self.axis_name,
      self.axis_index_groups,
      use_fast_variance=self.use_fast_variance,
      mask=mask,
    )
    mean = jnp.repeat(mean, self.group_size, axis=1)
    var = jnp.repeat(var, self.group_size, axis=1)
    return _normalize(
      x,
      mean,
      var,
      self.scale.value,
      self.bias.value,
      reduction_axes[:-1],
      (self.feature_axis,),
      self.dtype,
      self.epsilon,
    )