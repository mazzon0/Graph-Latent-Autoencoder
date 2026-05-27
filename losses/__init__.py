from .graph_latent_autoencoder_loss import GraphLatentAutoencoderLoss
from .gating_regularizers import ProbabilityRegularizer, DiscreteRegularizer
from .reconstruction_loss import L1Loss, L2Loss, BCELoss, SSIMLoss, HybridLoss

def get_loss(config: dict):
    recon_name = config.get('reconstruction', "mse")
    recon_config = config.get('reconstruction_' + recon_name, dict())

    print("Reconstruction Loss: ", end="")
    match config.get('reconstruction', "mse"):
        case 'mse' | 'l2':
            print("l2")
            reconstruction_loss = L2Loss(recon_config)
        case 'mae' | 'l1':
            print("l1")
            reconstruction_loss = L1Loss(recon_config)
        case 'ssim':
            print("ssim")
            reconstruction_loss = SSIMLoss(recon_config)
        case 'bce':
            print("bce")
            reconstruction_loss = BCELoss(recon_config)
        case 'hybrid':
            print("hybrid")
            reconstruction_loss = HybridLoss(recon_config)

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
    delay_epochs = config.get('delay_epochs', 0)
    ramp_epochs = config.get('ramp_epochs', 0)

    return GraphLatentAutoencoderLoss(reconstruction_loss, nodes_reg, edges_reg, alpha, beta, gamma, delay_epochs, ramp_epochs)
