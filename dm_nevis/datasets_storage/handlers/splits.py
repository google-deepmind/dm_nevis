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

"""Functions to manage splits.

Example of using the split function:

```
  per_split_gen = random_split_generator_into_splits_with_fractions(make_gen_fn,
  SPLIT_WITH_FRACTIONS_FOR_ALL_DATA)
```

"""

import functools
from typing import Any, Callable, Dict, Optional
from dm_nevis.datasets_storage.handlers import types
import numpy as np

SPLIT_WITH_FRACTIONS_FOR_ALL_DATA = {
    'train': 0.56,
    'dev': 0.12,
    'dev-test': 0.12,
    'test': 0.2
}
SPLIT_WITH_FRACTIONS_FOR_TRAIN = {'train': 0.7, 'dev': 0.15, 'dev-test': 0.15}
SPLIT_WITH_FRACTIONS_FOR_TRAIN_AND_DEV_ONLY = {'train': 0.8, 'dev': 0.2}
MERGED_TRAIN_AND_DEV = {'train_and_dev': ('train', 'dev')}
_DEFAULT_SPLIT_SEED = 0


# TODO: Make it producing more balanced random subsets.
# TODO: Add a parameter with total number of examples leading to
# better fractionning.
# TODO: Refactor it into splits.py file.
def random_split_generator_into_splits_with_fractions(
    make_gen_fn: Callable[[], types.DataGenerator],
    splits_with_fractions: Dict[str, float],
    merged_splits_to_split_names: Optional[Dict[str, Any]] = None,
    split_seed: int = _DEFAULT_SPLIT_SEED,
) -> Dict[str, types.DataGenerator]:
  """Randomly splits generator into disjoint subsets with specified fractions.

  The function goes sequentially through the elements of the original generator
  and randomly (based on categorical distribution with given fractions) assigns
  each element to a split. In order to create disjoint subsets, this function
  keeps an internal hash_map, which maps an id of the element from the original
  generator into assigned split.

  Args:
    make_gen_fn: Callable which creates a generator.
    splits_with_fractions: Dictionary with split_name into a resulting fraction.
    merged_splits_to_split_names: Optional dictionary mapping a new merged split
      names to original split names.
    split_seed: Seed used for random number generator in order to assign the
      fractions.

  Returns:
    Dictionary mapping split_name to a corresponding split generator.
  """
  fractions = [
      fraction for _, fraction in sorted(splits_with_fractions.items())
  ]
  assert np.isclose(np.sum(fractions), 1.0)
  random_state = np.random.RandomState(seed=split_seed)
  assign_fn = lambda x: np.argmax(random_state.multinomial(1, fractions))
  internal_hash_map = dict()

  def _hashed_select_from_assign_fn(x, expected_values_list, assign_fn,
                                    internal_hash_map):
    if x not in internal_hash_map:
      internal_hash_map[x] = assign_fn(x)
    return internal_hash_map[x] in expected_values_list

  def _select_subsplit(gen, select_fn):
    for idx, elem in enumerate(gen):
      if select_fn(idx):
        yield elem

  per_split_gen = dict()
  split_name_to_split_id = dict()
  for split_id, (split_name,
                 _) in enumerate(sorted(splits_with_fractions.items())):
    select_fn = functools.partial(
        _hashed_select_from_assign_fn,
        expected_values_list=[split_id],
        assign_fn=assign_fn,
        internal_hash_map=internal_hash_map)
    per_split_gen[split_name] = _select_subsplit(make_gen_fn(), select_fn)
    split_name_to_split_id[split_name] = split_id

  if merged_splits_to_split_names is None:
    return per_split_gen

  for (merged_split_name,
       splits_to_merge) in merged_splits_to_split_names.items():
    expected_values_list = []
    for split_name in splits_to_merge:
      if split_name not in splits_with_fractions:
        raise ValueError(
            f'{split_name} specified in `merged_splits_to_split_names` is not '
            'one of the original splits specified in `splits_with_fractions`.')
      expected_values_list.append(split_name_to_split_id[split_name])
    select_fn = functools.partial(
        _hashed_select_from_assign_fn,
        expected_values_list=expected_values_list,
        assign_fn=assign_fn,
        internal_hash_map=internal_hash_map)
    per_split_gen[merged_split_name] = _select_subsplit(make_gen_fn(),
                                                        select_fn)

  return per_split_gen
