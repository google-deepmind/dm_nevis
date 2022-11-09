# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model architectures."""

import inspect
from typing import Any, Callable, Iterable, Optional, Sequence, Type

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class FlattenOnly(hk.Module):
  """Null-model: Just flatten the image.

  The linear layer in the head will perform logistic regression.
  """

  def __call__(self, image: chex.Array, is_training: bool) -> chex.Array:
    del is_training
    return hk.Flatten()(image)


class MLP(hk.Module):
  """Flatten the image and apply a naive MLP with dropout."""

  def __init__(self,
               output_sizes: Sequence[int] = (4096, 4096),
               dropout_rate: Optional[float] = 0.5,
               name: Optional[str] = None):
    super().__init__(name)

    self._dropout_rate = dropout_rate
    self._model = hk.nets.MLP(output_sizes=output_sizes, activate_final=True)

  def __call__(self, image: chex.Array, is_training: bool) -> chex.Array:
    h = hk.Flatten()(image)

    if is_training and self._dropout_rate:
      h = self._model(h, dropout_rate=self._dropout_rate, rng=hk.next_rng_key())
    else:
      h = self._model(h)
    return h


class ConvNet(hk.Module):
  """A naive ConvNet."""

  def __init__(self, top_reduce: str = "mean", name: Optional[str] = None):
    super().__init__(name=name)
    self._top_reduce = top_reduce
    self._initial_conv = hk.Conv2D(128, [5, 5], stride=3)
    self._channels_per_stage = [
        [128, 128],
        [256, 256],
    ]

  def __call__(self, image: chex.Array, is_training: bool) -> chex.Array:
    del is_training

    h = self._initial_conv(image)
    h = jax.nn.relu(h)

    for stage in self._channels_per_stage:
      for channels in stage:
        h = hk.Conv2D(channels, [3, 3])(image)
        h = jax.nn.relu(h)
      h = hk.MaxPool([2, 2], [2, 2], padding="SAME", channel_axis=-1)(h)

    if self._top_reduce == "flatten":
      h = hk.Flatten()(h)
    elif self._top_reduce == "mean":
      h = jnp.mean(h, axis=[1, 2])
    elif self._top_reduce == "max":
      h = jnp.max(h, axis=[1, 2])
    else:
      raise ValueError(f"Unknown reduction-type {self._top_reduce}")

    return h


VGG_DEFAULT_CHANNELS = (64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512)
VGG_DEFAULT_STRIDES = (1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1)

VGG_4X_CHANNELS = tuple(4*c for c in VGG_DEFAULT_CHANNELS)


class VGG(hk.Module):
  """VGG Network with dropout, batchnorm and without maxpool."""

  def __init__(self,
               output_channels: Sequence[int] = VGG_DEFAULT_CHANNELS,
               strides: Sequence[int] = VGG_DEFAULT_STRIDES,
               dropout_rate: float = 0.0,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._dropout_rate = dropout_rate
    self._output_channels = output_channels
    self._strides = strides
    self._kernel_shapes = [[3, 3]] * len(self._output_channels)
    num_channels = len(self._output_channels)

    self._conv_modules = [
        hk.Conv2D(  # pylint: disable=g-complex-comprehension
            output_channels=self._output_channels[i],
            kernel_shape=self._kernel_shapes[i],
            stride=self._strides[i],
            name=f"conv_2d_{i}") for i in range(num_channels)
    ]
    self._bn_modules = [
        hk.BatchNorm(  # pylint: disable=g-complex-comprehension
            create_offset=True,
            create_scale=False,
            decay_rate=0.999,
            name=f"batch_norm_{i}") for i in range(num_channels)
    ]

  def __call__(self, inputs, is_training, test_local_stats=True):
    h = inputs
    for conv_layer, bn_layer in zip(self._conv_modules, self._bn_modules):
      h = conv_layer(h)
      h = bn_layer(
          h, is_training=is_training, test_local_stats=test_local_stats)
      if self._dropout_rate > 0 and is_training:
        h = hk.dropout(hk.next_rng_key(), rate=self._dropout_rate, x=h)
      h = jax.nn.relu(h)
    # Avg pool along axis 1 and 2
    h = jnp.mean(h, axis=[1, 2], keepdims=False, dtype=jnp.float64)
    return h


class DropConnect(hk.Module):
  """Batchwise Dropout used in EfficientNet."""

  def __init__(self, rate: float, name: Optional[str] = None):
    """Constructs a DropConnect module.

    Args:
      rate: Probability that each element of x is discarded. Must be a scalar in
        the range `[0, 1)`.
      name: (Optional) Name for this module.
    """
    super().__init__(name=name)
    self._rate = rate

  def __call__(self, x: chex.Array, is_training: bool) -> chex.Array:
    if not is_training:
      return x
    batch_size = x.shape[0]
    r = jax.random.uniform(
        hk.next_rng_key(), [batch_size] + [1] * (x.ndim - 1), dtype=x.dtype)
    keep_prob = 1. - self._rate
    binary_tensor = jnp.floor(keep_prob + r)
    return (x / keep_prob) * binary_tensor


class Dropout(hk.Module):
  """Dropout as a module."""

  def __init__(self, rate: float, name: Optional[str] = None):
    """Constructs a Dropout module.

    Args:
      rate: Probability that each element of x is discarded. Must be a scalar in
        the range `[0, 1)`.
      name: (Optional) Name for this module.
    """
    super().__init__(name=name)
    self._rate = rate

  def __call__(self, x: chex.Array, is_training: bool) -> chex.Array:
    if not is_training:
      return x
    return hk.dropout(hk.next_rng_key(), self._rate, x)


# We copy and adapt the attention component of Flax as Haiku's version does
# slightly different computations and prevents us from using pretrained
# checkpoints.
def _dot_product_attention(query, key, value, dtype=jnp.float32, axis=None):
  """Computes dot-product attention given query, key, and value."""
  assert key.shape[:-1] == value.shape[:-1]
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1])
  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
      raise ValueError("Attention axis must be between the batch "
                       "axis and the last-two axes.")

  depth = query.shape[-1]
  n = key.ndim
  # `batch_dims` is  <bs, <non-attention dims>, num_heads>.
  batch_dims = tuple(np.delete(range(n), axis + (n - 1,)))
  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels).
  qk_perm = batch_dims + axis + (n - 1,)
  key = key.transpose(qk_perm)
  query = query.transpose(qk_perm)
  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>).
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  query = query / jnp.sqrt(depth).astype(dtype)
  batch_dims_t = tuple(range(len(batch_dims)))
  attn_weights = jax.lax.dot_general(query, key, (((n - 1,), (n - 1,)),
                                                  (batch_dims_t, batch_dims_t)))

  # Normalize the attention weights.
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims)
  attn_weights = attn_weights.astype(dtype)

  # Compute the new values given the attention weights.
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = jax.lax.dot_general(attn_weights, value,
                          (wv_contracting_dims, (batch_dims_t, batch_dims_t)))

  # Back to (bs, dim1, dim2, ..., dimN, num_heads, channels).
  def _invert_perm(perm):
    perm_inv = [0] * len(perm)
    for i, j in enumerate(perm):
      perm_inv[j] = i
    return tuple(perm_inv)

  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  return y


# Adapted from `hk.MultiHeadAttention` but using the Flax attention function.
class MultiHeadAttention(hk.Module):
  """Multi-headed attention mechanism."""

  def __init__(self,
               num_heads: int,
               key_size: int,
               w_init: Optional[hk.initializers.Initializer] = None,
               query_size: Optional[int] = None,
               value_size: Optional[int] = None,
               model_size: Optional[int] = None,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.query_size = query_size or key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or key_size * num_heads
    if w_init is None:
      self.w_init = hk.initializers.VarianceScaling(1., "fan_avg", "uniform")
    else:
      self.w_init = w_init

  def __call__(self, query: chex.Array, key: chex.Array,
               value: chex.Array) -> chex.Array:
    """Compute MHA with queries, keys & values."""
    query_heads = self._linear_projection(query, self.query_size, "query")
    key_heads = self._linear_projection(key, self.key_size, "key")
    value_heads = self._linear_projection(value, self.value_size, "value")
    attention_vec = _dot_product_attention(
        query_heads, key_heads, value_heads, dtype=query.dtype, axis=1)
    attention_vec = jnp.reshape(attention_vec, (*query.shape[:2], -1))
    return hk.Linear(self.model_size, w_init=self.w_init)(attention_vec)

  @hk.transparent
  def _linear_projection(self,
                         x: chex.Array,
                         head_size: int,
                         name: Optional[str] = None) -> chex.Array:
    y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
    return y.reshape((*x.shape[:2], self.num_heads, head_size))


class SelfAttention(MultiHeadAttention):
  """Self-attention mechanism."""

  def __call__(self, x: chex.Array) -> chex.Array:
    return super().__call__(x, x, x)


def filter_kwargs(fn_or_class: Callable[..., Any]) -> Callable[..., Any]:
  """Argument cleaner for functions and class constructers."""
  method_fn = (fn_or_class.__init__ if isinstance(fn_or_class, Type) else
               fn_or_class)
  if isinstance(method_fn, hk.Module):
    # Haiku wraps `__call__` and destroys the `argspec`. However, it does
    # preserve the signature of the function.
    fn_args = list(inspect.signature(method_fn.__call__).parameters.keys())
  else:
    fn_args = inspect.getfullargspec(method_fn).args
  if fn_args and "self" == fn_args[0]:
    fn_args = fn_args[1:]
  def wrapper(*args, **kwargs):
    common_kwargs = {}
    if len(args) > len(fn_args):
      raise ValueError("Too many positional arguments.")
    for k, v in zip(fn_args, args):
      common_kwargs[k] = v
    for k, v in kwargs.items():
      if k in common_kwargs:
        raise ValueError(
            "{} already specified as a positional argument".format(k))
      if k in fn_args:
        common_kwargs[k] = v
    return fn_or_class(**common_kwargs)
  return wrapper
