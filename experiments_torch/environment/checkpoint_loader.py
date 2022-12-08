"""Functions for loading pretrained models from a checkpoint."""

from typing import Tuple, Iterator

from absl import logging
from experiments_torch.environment import pickle_checkpointer
from experiments_torch.training import models
from torch.nn import parameter


def load_ckpt_params(
    model: models.Model,
    freeze_pretrained_backbone: bool = False,
    checkpoint_path: str = '',
) -> Tuple[Iterator[parameter.Parameter]]:
  """Load pretrained model parameter from a checkpoint.

  Args:
    model: the model.
    freeze_pretrained_backbone: whether to freeze pretrained backbone or not.
    checkpoint_path: path to the pretrained checkpointer.

  Returns:
    updated params split into trainable and frozen.
  """

  checkpointer = pickle_checkpointer.PickleCheckpointer(checkpoint_path)
  restored_model = checkpointer.restore()

  if restored_model is None:
    return model.backbone.parameters(), {}

  assert isinstance(restored_model, models.Model)
  logging.info('Loading pretrained model finished.')

  for model_param, restored_model_param in zip(
      model.backbone.parameters(), restored_model.backbone.parameters()):
    assert model_param.data.shape == restored_model_param.data
    model_param.data = restored_model_param.data
    model_param.requires_grad = not freeze_pretrained_backbone

  if freeze_pretrained_backbone:
    return model.backbone.parameters(), model.heads_map.parameters()
  else:
    return model.parameters(), {}
