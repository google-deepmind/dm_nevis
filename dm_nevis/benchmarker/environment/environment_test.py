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

"""Tests for dm_nevis.benchmarker.environment.environment."""

import copy
from typing import Optional
from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.datasets import test_stream
from dm_nevis.benchmarker.environment import environment
from dm_nevis.benchmarker.learners import learner_interface
from dm_nevis.benchmarker.metrics import metrics_aggregators


class EnvironmentTest(parameterized.TestCase):

  def test_run(self):
    checkpointer = _InMemoryCheckpointer()
    stream = test_stream.TestStream()
    metrics = metrics_aggregators.noop_metrics_aggregator()
    learner = _build_test_learner()

    environment.run(
        learner, stream, metrics, write_checkpoint=checkpointer.write)
    self.assertNotEmpty(checkpointer.checkpoints)

    train_events = [
        event for event in stream.events()
        if isinstance(event, streams.TrainingEvent)
    ]

    for checkpoint in checkpointer.checkpoints:
      result = environment.run(
          learner, stream, metrics, checkpoint_to_resume=checkpoint)

      expected = {
          'seen_train_events': train_events,
          'values_of_x': [sum(range(20)), sum(range(20))],
      }

      self.assertEqual(expected, result.final_learner_state)


def _build_test_learner() -> learner_interface.Learner:

  def init():
    return {
        'seen_train_events': [],
        'values_of_x': [],
    }

  def train(event, state, write_checkpoint, *, checkpoint_to_resume=None):

    if checkpoint_to_resume:
      step, x, checkpoint_event = checkpoint_to_resume
      assert checkpoint_event == event
    else:
      x = step = 0

    for i in range(step, 20):
      if i % 3 == 0:
        write_checkpoint((i, x, event))
      x += i

    # Add to the learner state the value of x we computed, along with the
    # tain event that we used to compute it. In all cases, the value of x
    # will be the sum from 0 to 20, if checkpointing is working correctly!
    state = {
        'seen_train_events': [*state['seen_train_events'], event],
        'values_of_x': [*state['values_of_x'], x],
    }

    return state, learner_interface.ResourceUsage()

  def predict(event, state):
    del event, state
    return []

  return learner_interface.Learner(init, train, predict)


class _InMemoryCheckpointer:
  """A checkpointer that stores every checkpoint written in a list."""

  def __init__(self):
    self.checkpoints = []

  def write(self, ckpt: environment.EnvironmentState) -> None:
    self.checkpoints.append(copy.deepcopy(ckpt))

  def restore(self) -> Optional[environment.EnvironmentState]:
    if not self.checkpoints:
      return None

    return self.checkpoints[-1]

  def learner_checkpoints(self):
    return [ckpt.learner_state for ckpt in self.checkpoints]


if __name__ == '__main__':
  absltest.main()
