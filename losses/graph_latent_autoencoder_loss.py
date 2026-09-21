import torch
import torch.nn as nn
from utils.schedule import warmup_progress

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
        
        # Warmup scheduling (0 = dense graph, 1 = target sparsity)
        progress = warmup_progress(epoch, self.delay_epochs, self.ramp_epochs)

        # Sparsity evaluation
        losses['nodes'] = self.nodes_loss(outputs['node_conf'], outputs.get('node_keep'), progress)
        losses['edges'] = self.edges_loss(outputs['edge_conf'], outputs.get('edge_keep'), progress)

        # Number of elements kept after gating
        if outputs.get('node_keep') is not None:
            losses['nodes_kept'] = outputs['node_keep'].flatten(1).sum(dim=1).float().mean()
            losses['edges_kept'] = outputs['edge_keep'].flatten(1).sum(dim=1).float().mean()
            losses['node_threshold'] = outputs['node_threshold'].float()
            losses['edge_threshold'] = outputs['edge_threshold'].float()

        # Reconstruction evaluation
        for key, loss in self.reconstruction_loss(outputs['image'], targets, epoch).items():
            losses[key] = loss

        # Regularizers weight
        nodes_weight = 1.0 if self.nodes_loss.handles_warmup else progress
        edges_weight = 1.0 if self.edges_loss.handles_warmup else progress

        loss = (self.alpha * losses['reconstruction'] + self.beta * nodes_weight * losses['nodes'] + self.gamma * edges_weight * losses['edges'])
        
        losses['loss'] = loss
        losses['gating_warmup'] = torch.tensor(progress, dtype=torch.float32, device=targets.device)

        return losses