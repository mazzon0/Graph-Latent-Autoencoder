import torch.nn as nn
from .graph_latent_autoencoder_loss import GraphLatentAutoencoderLoss
from .gating_regularizers import ProbabilityRegularizer, DiscreteRegularizer
from .reconstruction_loss import SSIMLoss

def get_loss(config: dict):

    print("Reconstruction Loss: ", end="")
    match config.get('reconstruction', "mse"):
        case 'mse':
            print("mse")
            reconstruction_loss = nn.MSELoss()
        case 'mae':
            print("mae")
            reconstruction_loss = nn.L1Loss()
        case 'ssim':
            print("ssim")
            reconstruction_loss = SSIMLoss()
        case 'bce':
            print("bce")
            reconstruction_loss = nn.BCEWithLogitsLoss()

    print("Nodes Regularizer: ", end="")
    match config.get('nodes', "probs"):
        case 'probs':
            print("probs")
            nodes_reg = ProbabilityRegularizer()
        case 'discr':
            print("discr")
            nodes_reg = DiscreteRegularizer(0.5)

    print("Edges Regularizer: ", end="")
    match config.get('edges', "probs"):
        case 'probs':
            print("probs")
            edges_reg = ProbabilityRegularizer()
        case 'discr':
            print("discr")
            edges_reg = DiscreteRegularizer(0.5)

    alpha = config.get('alpha', 1.0)
    beta  = config.get('beta', 1.0)
    gamma = config.get('gamma', 1.0)

    return GraphLatentAutoencoderLoss(reconstruction_loss, nodes_reg, edges_reg, alpha, beta, gamma)
