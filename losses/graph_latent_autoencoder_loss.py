import torch
import torch.nn as nn

class GraphLatentAutoencoderLoss(nn.Module):
    def __init__(self, reconstruction_loss: nn.Module, nodes_loss: nn.Module, edges_loss: nn.Module, alpha: float, beta: float, gamma: float):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.nodes_loss = nodes_loss
        self.edges_loss = edges_loss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs: dict, targets: torch.Tensor, epoch: int):
        losses = dict()
        losses['nodes'] = self.nodes_loss(outputs['node_conf'])
        losses['edges'] = self.edges_loss(outputs['edge_conf'])

        for key, loss in self.reconstruction_loss(outputs['image'], targets, epoch).items():
            losses[key] = loss

        loss = self.alpha * losses['reconstruction'] + self.beta * losses['nodes'] + self.gamma * losses['edges']
        losses['loss'] = loss
        return losses