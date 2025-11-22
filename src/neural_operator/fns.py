import torch

from box import Box
from torch import nn
from .base import NeuralOperatorBase


class FNS1d(NeuralOperatorBase):
    def __init__(self, config: Box):
        super().__init__(config)
        pass
