import torch
import torch.nn as nn

class GraphLatentAutoencoderLoss(nn.Module):
    def __init__(self, reconstruction_loss: nn.Module, nodes_loss: torch.Tensor, edges_loss: torch.Tensor, alpha: float, beta: float, gamma: float):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.nodes_loss = nodes_loss
        self.edges_loss = edges_loss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs, node_probs, edge_probs, targets, epoch):
        losses = dict()
        losses['nodes'] = self.nodes_loss(node_probs)
        losses['edges'] = self.edges_loss(edge_probs)

        for key, loss in self.reconstruction_loss(outputs, targets, epoch).items():
            losses[key] = loss

        loss = self.alpha * losses['reconstruction'] + self.beta * losses['nodes'] + self.gamma * losses['edges']
        losses['loss'] = loss

        return losses
