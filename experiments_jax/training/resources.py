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

"""A module for estimating resource usage of jax functions."""

from typing import Any, Callable
import jax


def estimate_flops(fn: Callable[..., Any], *args, **kwargs) -> float:
  """Runs the given jitable JAX function with parameters and return #FLOPs."""
  xe = jax.lib.xla_client._xla  # pylint: disable=protected-access
  xla_backend = jax.lib.xla_bridge.get_backend("cpu")

  static_argnums = kwargs.pop("static_argnums", ())
  c = jax.xla_computation(fn, static_argnums=static_argnums)(*args, **kwargs)
  e = xla_backend.compile(c)
  m, = e.hlo_modules()
  analysis = xe.hlo_module_cost_analysis(xla_backend, m)
  return float(analysis["flops"])
