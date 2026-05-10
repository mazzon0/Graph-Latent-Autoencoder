import math
import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from abc import ABC, abstractmethod

class LossModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, outputs, targets, epoch=0):
        """Should return the loss"""
        pass

class L1Loss(LossModule):
    """L1 Loss (Mean Absolute Error)"""
    def __init__(self, config: dict):
        super().__init__()
        reduction = config.get('reduction', "mean")
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, outputs, targets, epoch=0):
        return {'reconstruction': self.loss(outputs, targets)}

class L2Loss(LossModule):
    """L2 Loss (Mean Squared Error)"""
    def __init__(self, config: dict):
        super().__init__()
        reduction = config.get('reduction', "mean")
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, outputs, targets, epoch=0):
        return {'reconstruction': self.loss(outputs, targets)}
    
class BCELoss(LossModule):
    """Binary Cross Emtropy Loss"""
    def __init__(self, config: dict):
        super().__init__()
        reduction = config.get('reduction', "mean")
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, outputs, targets):
        return {'reconstruction': self.loss(outputs, targets)}

class SSIMLoss(LossModule):
    """Structural Similarity Index Measure Loss"""
    def __init__(self, config: dict):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, outputs, targets):
        return {'reconstruction': (1 - self.ssim(outputs, targets)) / 2}
    
class HybridLoss(LossModule):
    """Hybrid Loss between SSIM and L1"""
    def __init__(self, config: dict):
        super().__init__()
        self.start_val = float(config.get('start_val', 0.5))
        self.end_val = float(config.get('end_val', 0.5))
        self.start_epoch = int(config.get('start_epoch', 0))
        self.end_epoch = int(config.get('end_epoch', 10))
        self.func = config.get('func', "linear")

        assert(self.start_val <= self.end_val)
        assert(self.start_epoch <= self.end_epoch)

        self.ssim = SSIMLoss(dict())
        self.mae  = nn.L1Loss()

    def forward(self, outputs, targets, epoch=0):
        if epoch <= self.start_epoch:
            alpha = self.start_val

        elif epoch < self.end_epoch:
            x = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            match self.func:
                case "linear":
                    pass
                case "cosine":
                    x = 1 - math.cos(x)

            alpha = x * self.end_val + (1 - x) * self.start_val

        else:
            alpha = self.end_val

        mae = self.mae(outputs, targets)
        ssim = self.ssim(outputs, targets)['reconstruction']
        return {
            'reconstruction': (1 - alpha) * mae + alpha * ssim,
            'l1': mae,
            'ssim': ssim,
            'alpha': torch.Tensor([alpha])
        }