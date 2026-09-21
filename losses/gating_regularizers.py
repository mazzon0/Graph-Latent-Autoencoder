import torch
import torch.nn as nn
import torch.nn.functional as F

# All the regularizers receive the confidences conf, the boolean mask keep of the elements that survived the gating
# of the model and the progress of the sparsity schedule (0.0 = dense graph, 1.0 = target sparsity).
# handles_warmup tells if the regularizer applies the schedule by itself, otherwise the loss scales its weight by progress.

class ProbabilityRegularizer(nn.Module):
    handles_warmup = False

    def __init__(self):
        super().__init__()

    def forward(self, conf: torch.Tensor, keep: torch.Tensor = None, progress: float = 1.0):
        return conf.mean() 
    
class DiscreteRegularizer(nn.Module):
    handles_warmup = False

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, conf: torch.Tensor, keep: torch.Tensor = None, progress: float = 1.0):
        hard = (conf > self.threshold).float()
        active = conf * hard
        return (hard + active - active.detach()).mean()

class BandRegularizer(nn.Module):
    handles_warmup = True

    def __init__(self, config: dict):
        super().__init__()
        assert 'min' in config and 'max' in config, "the band regularizer needs 'min' and 'max'"
        self.min = float(config['min'])
        self.max = float(config['max'])
        assert 0 <= self.min <= self.max and self.max > 0

    def forward(self, conf: torch.Tensor, keep: torch.Tensor = None, progress: float = 1.0):
        if conf.dim() < 2:  # models without a latent graph are ignored
            return conf.new_zeros(())

        B = conf.shape[0]
        conf = conf.reshape(B, -1)
        M = conf.shape[1]
        hard = keep.reshape(B, -1).float() if keep is not None else (conf > 0.5).float()
        count = (hard + conf - conf.detach()).sum(dim=1)

        low = min(self.min, M) * progress
        high = M + (min(self.max, M) - M) * progress
        excess = F.relu(low - count) + F.relu(count - high)
        return (excess / self.max).mean()
