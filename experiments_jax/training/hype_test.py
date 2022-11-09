"""Tests for hype."""

from absl.testing import absltest
import chex
from experiments_jax.training import hype


class HypeTest(absltest.TestCase):

  def test_search_space_consistent(self):
    """Test that the search space generated is deterministic."""

    expected = [{
        'lr': 0.0002529,
        'ls': 0.2868102
    }, {
        'lr': 0.0348578,
        'ls': 0.2843482
    }, {
        'lr': 0.0195579,
        'ls': 0.0169654
    }, {
        'lr': 0.0005823,
        'ls': 0.0254615
    }, {
        'lr': 0.0030641,
        'ls': 0.2506496
    }, {
        'lr': 0.0022308,
        'ls': 0.2207909
    }, {
        'lr': 0.0090111,
        'ls': 0.2009191
    }]

    sweep = hype.zipit(
        hype.sweep(
            'lr',
            hype.log_uniform_random_values(1e-4, 1e-1, seed=1, n=7),
        ), hype.sweep(
            'ls',
            hype.uniform_random_values(0.0, 0.3, seed=2, n=7),
        ))

    chex.assert_trees_all_close(list(sweep), expected, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
