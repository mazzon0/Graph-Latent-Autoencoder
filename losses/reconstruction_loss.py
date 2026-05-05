import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, outputs, targets):
        return (1 - self.ssim(outputs, targets)) / 2