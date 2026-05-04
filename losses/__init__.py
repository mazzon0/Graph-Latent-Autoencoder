import torch.nn as nn
from .graph_latent_autoencoder_loss import GraphLatentAutoencoderLoss
from .gating_regularizers import ProbabilityRegularizer, DiscreteRegularizer
from .reconstruction_loss import SSIMLoss

def get_loss(config: dict):

    match config.get('reconstruction', "mse"):
        case 'mse':
            reconstruction_loss = nn.MSELoss()
        case 'ssim':
            reconstruction_loss = SSIMLoss()

    match config.get('nodes', "probs"):
        case 'probs':
            nodes_reg = ProbabilityRegularizer()
        case 'discr':
            nodes_reg = DiscreteRegularizer(0.5)

    match config.get('edges', "probs"):
        case 'probs':
            edges_reg = ProbabilityRegularizer()
        case 'discr':
            edges_reg = DiscreteRegularizer(0.5)

    alpha = config.get('alpha', 1.0)
    beta  = config.get('beta', 1.0)
    gamma = config.get('gamma', 1.0)

    return GraphLatentAutoencoderLoss(reconstruction_loss, nodes_reg, edges_reg, alpha, beta, gamma)
