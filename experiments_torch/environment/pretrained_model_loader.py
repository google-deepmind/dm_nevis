"""Functions for loading pretrained models from a checkpoint."""

from typing import Tuple, Union, Dict

from absl import logging
from experiments_torch.training import models
from experiments_torch.training import trainer
from torch.nn import parameter


def load_model_params_from_ckpt(
    model: models.Model,
    freeze_pretrained_backbone: bool = False,
    checkpoint_path: str = '',
) -> Tuple[parameter.Parameter, Union[parameter.Parameter, Dict]]:  # pylint: disable=g-bare-generic
  """Load pretrained model parameter from a checkpoint.

  Args:
    model: the model.
    freeze_pretrained_backbone: whether to freeze pretrained backbone or not.
    checkpoint_path: path to the pretrained checkpointer.

  Returns:
    updated params split into trainable and frozen.
  """
  trainer_state = trainer.restore_train_state(checkpoint_path)
  if trainer_state is None or trainer_state.model is None:
    return model.backbone.parameters(), {}

  restored_model = trainer_state.model

  assert isinstance(restored_model, models.Model)
  logging.info('Loading pretrained model finished.')

  for model_param, restored_model_param in zip(
      model.backbone.parameters(), restored_model.backbone.parameters()):
    assert model_param.data.shape == restored_model_param.data.shape
    model_param.data = restored_model_param.data
    model_param.requires_grad = not freeze_pretrained_backbone

  if freeze_pretrained_backbone:
    return model.backbone.parameters(), model.heads_map.parameters()
  else:
    return model.parameters(), {}
