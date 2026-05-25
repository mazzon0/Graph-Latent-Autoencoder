import torch
import torch.nn as nn

class ProbabilityRegularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, conf: torch.Tensor):
        return conf.mean() 
    
class DiscreteRegularizer(nn.Module):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, conf: torch.Tensor):
        return (conf > self.threshold).float().mean()