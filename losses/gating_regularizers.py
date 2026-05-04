import torch
import torch.nn as nn

class ProbabilityRegularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs: torch.Tensor):
        return probs.sum()
    
class DiscreteRegularizer(nn.Module):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, decisions: torch.Tensor):
        return (decisions > self.threshold).sum()