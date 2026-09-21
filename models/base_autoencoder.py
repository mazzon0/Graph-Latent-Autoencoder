from abc import ABC, abstractmethod
import torch.nn as nn

class BaseAutoencoder(nn.Module, ABC):
    @abstractmethod
    def get_first_layer(self) -> nn.Module:
        """Should return the first learnable layer"""
        pass

    def set_sparsity_progress(self, progress: float):
        """
        Sets the progress of the sparsity schedule (0 = dense graph, 1 = target sparsity).
        Models without a latent graph ignore it.
        """
        pass