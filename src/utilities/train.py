import abc
import dataclasses
import random
import sys
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
# noinspection PyUnresolvedReferences,PyProtectedMember
from torch.optim.lr_scheduler import _LRScheduler, SequentialLR


class IPretrainer(metaclass=abc.ABCMeta):
    # @abc.abstractmethod
    # def net(self) -> nn.Module:
    #     raise NotImplementedError
    ...


@dataclasses.dataclass
class TrainingSetup:
    optimizer: Optimizer or None = None
    loss_function: nn.Module or None = None
    lr_scheduler: _LRScheduler or None = None
    num_epochs: int = 1
    seed: int or None = None
    device: str = 'cpu'
    train_config: dict or None = None

    def __post_init__(self):
        super().__init__()
        if self.seed is not None:
            self.apply_seed()

    @property
    def scheduler(self):
        return self.lr_scheduler

    @scheduler.setter
    def scheduler(self, scheduler):
        self.lr_scheduler = scheduler

    @property
    def optim(self):
        return self.optimizer

    @optim.setter
    def optim(self, optim):
        self.optimizer = optim

    @property
    def loss(self):
        return self.loss_function

    @loss.setter
    def loss(self, loss):
        self.loss_function = loss

    def apply_seed(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f'\t[TrainingSetup::apply_seed] Seed set to {seed}')

    def load_state(self, state_dict: dict) -> List[str]:
        loaded_keys = []
        for key in ['optimizer', 'lr_scheduler']:
            if key in state_dict.keys():
                try:
                    getattr(self, key).load_state_dict(state_dict.pop(key))
                except ValueError as e:
                    print(f'\t[TrainingSetup::tate] Error loading "{key}": {str(e)}', file=sys.stderr)
                    break
            loaded_keys.append(key)
        return loaded_keys

    def state_dict(self) -> Dict[str, dict]:
        return {key: getattr(self, key).state_dict() for key in ['optimizer', 'lr_scheduler']}

    def reset_optimizer(self, new_model=None, reset_lrs: bool = True) -> Optimizer or Tuple[Optimizer, _LRScheduler]:
        params = new_model.parameters() if new_model is not None else self.optimizer.param_groups[0]['params']
        self.optimizer = self.optimizer.__class__(params, **{k: v for k, v in self.optimizer.defaults.items() if
                                                             k in ['lr', 'betas', 'weight_decay']})
        if reset_lrs and self.scheduler is not None:
            assert self.train_config is not None, 'Cannot re-instantiate LR Scheduler unless given the YAML-like config'
            from utilities.config import ConfigReader
            resolved, _ = ConfigReader.from_config(self.train_config['scheduler'],
                                                   deferred_args=dict(optimizer=self.optimizer))
            self.scheduler = resolved.popitem(last=True)[1]
