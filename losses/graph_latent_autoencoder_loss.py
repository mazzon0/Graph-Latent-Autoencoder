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

    def forward(self, outputs, node_probs, edge_probs, targets):
        nodes = self.nodes_loss(node_probs)
        edges = self.edges_loss(edge_probs)
        reconstruction = self.reconstruction_loss(outputs, targets)
        loss = self.alpha * reconstruction + self.beta * nodes + self.gamma * edges

        return {
            'loss': loss,
            'reconstruction': reconstruction,
            'nodes': nodes,
            'edges': edges
        }
