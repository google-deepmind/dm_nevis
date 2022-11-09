"""Support creation of search spaces for hyperparameter searches."""

import math
import random
import typing
from typing import Any, Sized, Iterable, Protocol, TypeVar

T = TypeVar('T')


@typing.runtime_checkable
class SizedIterable(Sized, Iterable[T], Protocol[T]):
  """A type annotation to represent types with __len__ and __iter__ methods.

  Types satisfying this interface may be iterated over, to yield a fixed number
  of elements of type T. The total number of elements yieled is equal to value
  returned by __len__ for the type. There is no requirement that the yielded
  elements be identical on each iteration, and this property is useful for
  representing sequences of values sampled from a probability distribution.

  >>> x: SizedIterable[int] = [1, 2, 3]
  >>> isinstance(x, SizedIterable)
  True
  """


Value = Any
Overrides = dict[str, Value]
Sweep = SizedIterable[Overrides]


def sweep(key: str, values: SizedIterable[Value]) -> Sweep:
  """Combines a key with values to create a sweep.

  >>> list(sweep('a', [1, 2, 3]))
  [{'a': 1}, {'a': 2}, {'a': 3}]

  Each time the sweep is iterated, the sweep is re-generated from the values.
  This means each subsequent iteration of the sweep can yield different values,
  if the underlying values are from a randomized source. This makes it possible
  to build search spaces by producting together sweeps, but requires some care
  to ensure determinism.

  Args:
    key: The key that this sweep should override with values.
    values: The iterable values to apply to the key.

  Returns:
    An iterable, sized object which may be iterated to yield overrides.
  """

  class ValueSweep:

    def __len__(self):
      return len(values)

    def __iter__(self):
      yield from ({key: value} for value in values)

  return ValueSweep()


def chain(*sweeps: Sweep) -> Sweep:
  """Chains sweeps together.

  >>> list(chain([{'a': 1}, {'a': 2}], [{'b': 1}, {'b': 2}]))
  [{'a': 1}, {'a': 2}, {'b': 1}, {'b': 2}]

  Args:
    *sweeps: A sequence of sweeps.

  Returns:
    A new sweep which, on each iteration, iterates over the given sweeps in
    order.
  """

  class ChainSweep:

    def __len__(self):
      return sum(len(s) for s in sweeps)

    def __iter__(self):
      for s in sweeps:
        yield from s

  return ChainSweep()


def zipit(*sweeps: Sweep) -> Sweep:
  """Zips sweeps together.

  >>> list(zipit([{'a': 1}, {'a': 2}], [{'b': 1}, {'b': 2}]))
  [{'a': 1, 'b': 1}, {'a': 2, 'b': 2}]

  Args:
    *sweeps: A sequence of sweeps.

  Returns:
    A new sweep which, on each iteration, will iterate all of the given sweeps
    together and combine their overrides.
  """

  lengths = set(len(s) for s in sweeps)
  if len(lengths) != 1:
    msg = ', '.join(f'{type(s)}: {len(s)}' for s in sweeps)
    raise ValueError(f'zip expects sweeps to have same length. Got {msg}')

  class ZipSweep:
    """A collection of sweeps combined using zipit."""

    def __len__(self):
      return len(sweeps[0])

    def __iter__(self):
      iters = [iter(s) for s in sweeps]

      for _ in range(len(sweeps[0])):
        result = {}
        for it in iters:
          result.update(next(it))
        yield result

  return ZipSweep()


def product(*sweeps: Sweep) -> Sweep:
  """Lazily generates the Cartesian product over sweeps.

  >>> list(product([{'a': 1}, {'a': 2}], [{'b': 1}, {'b': 2}]))
  [{'a': 0, 'b': 0}, {'a': 0, 'b': 1}, {'a': 1, 'b': 0}, {'a': 1, 'b': 1}]

  For the product over sweeps s0, s1, containing non-randomly-sampled values,
  the behaviour is identical to

  >>> ({**a, **b} for a, b in itertools.product(s0, s1))

  If the sweeps contain randomly sampled value ranges however, product will
  guarantee that every sample returned is uniquely drawn from the underlying
  sweeps.

  Args:
    *sweeps: A sequence of sweeps.

  Returns:
    A new sweep which, on each iteration, computes the product of the given
    sweeps. We guarantee that every value in the returned sweep is used exactly
    once in the returned sweep. This means that it is safe to used product
    with randomly sampled sweeps, and values will not be duplicated across
    the overrides. This is a key difference with, for example,
    itertools.product, or hyper.
  """

  class ProductSweep:
    """A collection of sweeps combined using product."""

    def __len__(self):
      return math.prod(len(s) for s in sweeps)

    def __iter__(self):
      if not sweeps:
        yield from []
      elif len(sweeps) == 1:
        yield from sweeps[0]
      elif len(sweeps) > 2:
        yield from product(product(sweeps[0], sweeps[1]), *sweeps[2:])
      else:

        # Note that it would feel more natural to use the following code below,
        #
        # for a in sweeps[0]:
        #   for b in sweeps[1]:
        #       yield {**a, **b}
        #
        # However, this would introduce a subtle bug: sweeps[1] would be
        # iterated len(sweeps[1]) times, and sweeps[0] would only be iterated
        # once, with each yielded value being reused several times in the inner
        # loop. Since sweeps may be over randomly sampled ranges, this can cause
        # the distribution of the returned values to collapse, resulting in
        # repeated values that should have been drawn independently at random.
        #
        # To solve this, we maintain a cache of iterators over the outer sweep,
        # with one iterator for each element of the inner sweep. Each time we
        # must sample from the outer iterator in the inner loop, we advance the
        # appropriate cached iterator.
        #
        # As a further optimization, the iterators are lazily constructed on
        # first access, to allow us to efficiently iterate over the first few
        # elements of very large products.
        #
        # This solution is optimal in the number of samples drawn from the
        # iterators, and uses memory proportional to the length of the the inner
        # iterator. Furthermore, the order of iteration of the values is
        # identical to the naive loop implementation, for the case of iterables
        # that always yield the same values (such as regular sequences).

        iters = []

        for i in range(len(sweeps[0])):
          for j, dct in enumerate(sweeps[1]):

            if i == 0:
              iters.append(iter(sweeps[0]))

            yield {**next(iters[j]), **dct}

  return ProductSweep()


def log_uniform_random_values(lo: float, hi: float, *, n: int,
                              seed: int) -> SizedIterable[float]:
  """Returns an iterable that yields log-uniform distributed values.

  >>> list(log_uniform_random_values(1e-1, 1e-3, n=3, seed=0))
  [0.0020471812581430546, 0.003048535060583298, 0.014416400482129478]

  Args:
    lo: The lower-bound of the log-uniform random number generator.
    hi: The upper-bound of the log-uniform random number generator.
    n: The number of values returned during a single iteration over the values.
    seed: A value to seed the random number generator.

  Returns:
    A collection of floats sampled from a log-uniform distribution [lo, hi].
    Note that each iteration over the values will be different from the last,
    but that the values are always deterministic.
  """
  values = uniform_random_values(math.log(lo), math.log(hi), n=n, seed=seed)

  class LogUniformRandomValues:

    def __len__(self):
      return n

    def __iter__(self):
      return (math.exp(v) for v in values)

  return LogUniformRandomValues()


def uniform_random_values(lo: float, hi: float, *, n: int,
                          seed: int) -> SizedIterable[float]:
  """Returns an iterable that yields uniform distributed values.

  >>> list(uniform_random_values(0, 1, n=3, seed=0))
  [0.8444218515250481, 0.7579544029403025, 0.420571580830845]

  Args:
    lo: The lower-bound of the uniform random number generator.
    hi: The upper-bound of the uniform random number generator.
    n: The number of values returned during a single iteration over the values.
    seed: A value to seed the random number generator.

  Returns:
    A collection of floats sampled from a uniform distribution [lo, hi].
    Note that each iteration over the values will be different from the last,
    but that the values are always deterministic.
  """
  rng = random.Random(seed)

  class UniformRandomValues:

    def __len__(self):
      return n

    def __iter__(self):
      return (rng.uniform(lo, hi) for _ in range(n))

  return UniformRandomValues()
