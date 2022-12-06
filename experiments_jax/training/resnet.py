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

"""Implements customized ResNets that return embeddings and logits."""

from typing import Dict, Mapping, Optional, Sequence, Union
import haiku as hk
import jax
import jax.numpy as jnp

FloatStrOrBool = Union[str, float, bool]


def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class ResNet(hk.nets.ResNet):
  """Original Resnet model that returns embeddings."""

  def __init__(self, num_classes=10, **kwargs):
    # haiku expects num_classes, but the top layer won't be instantiated.
    super().__init__(num_classes=num_classes, **kwargs)

  def __call__(self,
               inputs: jnp.ndarray,
               is_training: bool,
               test_local_stats: bool = False) -> Dict[str, jnp.ndarray]:
    out = inputs
    out = self.initial_conv(out)
    if not self.resnet_v2:
      out = self.initial_batchnorm(out, is_training, test_local_stats)
      out = jax.nn.relu(out)

    out = hk.max_pool(out,
                      window_shape=(1, 3, 3, 1),
                      strides=(1, 2, 2, 1),
                      padding="SAME")

    for block_group in self.block_groups:
      out = block_group(out, is_training, test_local_stats)

    if self.resnet_v2:
      out = self.final_batchnorm(out, is_training, test_local_stats)
      out = jax.nn.relu(out)
    out = jnp.mean(out, axis=(1, 2))
    return out


class ResNet18(ResNet):
  """ResNet18 model."""

  def __init__(
      self,
      bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      strides: Sequence[int] = (1, 2, 2, 2),
  ):
    """Constructs a ResNet18 model that returns both embeddings and logits.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
      strides: A sequence of length 4 that indicates the size of stride of
        convolutions for each block in each group.
    """
    super().__init__(bn_config=bn_config,
                     initial_conv_config=initial_conv_config,
                     resnet_v2=resnet_v2,
                     strides=strides,
                     name=name,
                     **hk.nets.ResNet.CONFIGS[18])


class ResNet50(ResNet):
  """ResNet50 model."""

  def __init__(
      self,
      bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      strides: Sequence[int] = (1, 2, 2, 2),
  ):
    """Constructs a ResNet50 model (hk.ResNet18 that returns both embeddings and logits).

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
      strides: A sequence of length 4 that indicates the size of stride
        of convolutions for each block in each group.
    """
    super().__init__(num_classes=1,  # fake head
                     bn_config=bn_config,
                     initial_conv_config=initial_conv_config,
                     resnet_v2=resnet_v2,
                     strides=strides,
                     name=name,
                     **hk.nets.ResNet.CONFIGS[50])


class CifarResNet(hk.Module):
  """ResNet model for CIFAR10 (almost) following original ResNet paper.

  This is different from the haiku.nets.Resnet implementation in two ways:
  1. Initital convolution is 3x3 with stride 1 (instead of 7x7 with stride 2),
  2. No max-pool before the block groups.
  Note: the haiku.nets.Resnet implementation fits larger inputs better
  (e.g., Imagenet).

  Original ResNet paper (arxiv:1512.03385) allows uses fewer channels but we do
  not implement that here.
  """

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      bn_config: Optional[Mapping[str, float]] = None,
      resnet_v2: bool = False,
      bottleneck: bool = True,
      channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
      use_projection: Sequence[bool] = (True, True, True, True),
      name: Optional[str] = None,
  ):
    """Constructs a ResNet model.

    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
      bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
      channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.resnet_v2 = resnet_v2

    bn_config = dict(bn_config or {})
    bn_config.setdefault("decay_rate", 0.9)
    bn_config.setdefault("eps", 1e-5)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)

    # Number of blocks in each group for ResNet.
    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")

    self.initial_conv = hk.Conv2D(
        output_channels=64,
        kernel_shape=3,
        stride=1,
        with_bias=False,
        padding="SAME",
        name="initial_conv")

    if not self.resnet_v2:
      self.initial_batchnorm = hk.BatchNorm(name="initial_batchnorm",
                                            **bn_config)

    self.block_groups = []
    strides = (1, 2, 2, 2)
    for i in range(4):
      self.block_groups.append(
          hk.nets.ResNet.BlockGroup(channels=channels_per_group[i],
                                    num_blocks=blocks_per_group[i],
                                    stride=strides[i],
                                    bn_config=bn_config,
                                    resnet_v2=resnet_v2,
                                    bottleneck=bottleneck,
                                    use_projection=use_projection[i],
                                    name="block_group_%d" % (i)))

    if self.resnet_v2:
      self.final_batchnorm = hk.BatchNorm(name="final_batchnorm", **bn_config)

  def __call__(self,
               inputs: jnp.ndarray,
               is_training: bool,
               test_local_stats: bool = False) -> Dict[str, jnp.ndarray]:
    out = inputs
    out = self.initial_conv(out)
    if not self.resnet_v2:
      out = self.initial_batchnorm(out, is_training, test_local_stats)
      out = jax.nn.relu(out)

    for block_group in self.block_groups:
      out = block_group(out, is_training, test_local_stats)

    if self.resnet_v2:
      out = self.final_batchnorm(out, is_training, test_local_stats)
      out = jax.nn.relu(out)
    out = jnp.mean(out, axis=(1, 2))
    return out


class CifarResNet18(CifarResNet):
  """CifarResNet18."""

  def __init__(self,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(2, 2, 2, 2),
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=False,
                     channels_per_group=(64, 128, 256, 512),
                     use_projection=(False, True, True, True),
                     name=name)


class CifarResNet34(CifarResNet):
  """CifarResNet34."""

  def __init__(self,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      name: Name of the model.
    """
    super().__init__(blocks_per_group=(3, 4, 6, 3),
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=False,
                     channels_per_group=(64, 128, 256, 512),
                     use_projection=(False, True, True, True),
                     name=name)


class CifarResNet50(CifarResNet):
  """CifarResNet50."""

  def __init__(self,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 4, 6, 3),
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=True,
                     name=name)


class CifarResNet101(CifarResNet):
  """CifarResNet101."""

  def __init__(self,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 4, 23, 3),
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=True,
                     name=name)


class CifarResNet152(CifarResNet):
  """CifarResNet152."""

  def __init__(self,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 8, 36, 3),
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=True,
                     name=name)


class CifarResNet200(CifarResNet):
  """CifarResNet200."""

  def __init__(self,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 24, 36, 3),
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=True,
                     name=name)
