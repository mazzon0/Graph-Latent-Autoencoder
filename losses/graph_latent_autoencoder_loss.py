import torch
import torch.nn as nn

class GraphLatentAutoencoderLoss(nn.Module):
    def __init__(self, reconstruction_loss: nn.Module, nodes_loss: nn.Module, edges_loss: nn.Module, alpha: float, beta: float, gamma: float, delay_epochs: int = 10, ramp_epochs: int = 10):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.nodes_loss = nodes_loss
        self.edges_loss = edges_loss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delay_epochs = delay_epochs
        self.ramp_epochs = ramp_epochs

    def forward(self, outputs: dict, targets: torch.Tensor, epoch: int):
        losses = dict()
        
        # Sparsity evaluation
        losses['nodes'] = self.nodes_loss(outputs['node_conf'])
        losses['edges'] = self.edges_loss(outputs['edge_conf'])

        # Reconstruction evaluation
        for key, loss in self.reconstruction_loss(outputs['image'], targets, epoch).items():
            losses[key] = loss

        # Warmup scheduling
        if epoch < self.delay_epochs:
            warmup_factor = 0.0
        else:
            warmup_factor = min(1.0, (epoch - self.delay_epochs) / self.ramp_epochs)

        loss = (self.alpha * losses['reconstruction'] + 
                warmup_factor * (self.beta * losses['nodes'] + self.gamma * losses['edges']))
        
        losses['loss'] = loss
        losses['gating_warmup'] = torch.tensor(warmup_factor, dtype=torch.float32, device=targets.device)

        return losses