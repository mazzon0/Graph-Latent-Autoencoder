import torch.nn as nn

def get_loss(name: str, config: dict):

    match(name):
        case 'mse':
            return nn.MSELoss() # TODO wrap and return a dict

    return None