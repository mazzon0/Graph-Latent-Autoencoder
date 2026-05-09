from abc import ABC, abstractmethod
import torch.nn as nn

class BaseAutoencoder(nn.Module, ABC):
    @abstractmethod
    def get_first_layer(self) -> nn.Module:
        """Should return the first learnable layer"""
        pass