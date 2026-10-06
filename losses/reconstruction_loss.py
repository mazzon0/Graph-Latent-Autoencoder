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

    def forward(self, outputs, targets, epoch=0):
        return {'reconstruction': self.loss(outputs, targets)}

class SSIMLoss(LossModule):
    """
    Structural Similarity Index Measure Loss.

    SSIM compares the two images over a sliding window, and the size of that window sets the scale of the detail
    it can see. Which parameter controls it depends on the kind of window (measured on torchmetrics 1.9):

        gaussian_kernel: true  (the default)  ->  `sigma` is the only knob, `kernel_size` is ignored.
                                                  sigma 1.5 gives an effective window of about 7x7.
        gaussian_kernel: false (uniform box)  ->  `kernel_size` is the knob.

    A smaller window punishes blur and small-object errors more. On 64x64 CLEVR, against a blurred target:
    gaussian sigma 1.5 -> 0.908, sigma 1.0 -> 0.901; uniform 11 -> 0.924, uniform 7 -> 0.912, uniform 5 -> 0.902.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0,
            gaussian_kernel=bool(config.get('gaussian_kernel', True)),
            kernel_size=int(config.get('kernel_size', 11)),
            sigma=float(config.get('sigma', 1.5)))

    def forward(self, outputs, targets, epoch=0):
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

        self.ssim = SSIMLoss(config)    # so that kernel_size and sigma can be set on the hybrid too
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
                    x = (1 - math.cos(math.pi * x)) / 2

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