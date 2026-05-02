import torch.nn as nn

def get_loss(name: str, config: dict):

    match(name):
        case 'cross_entropy':
            return nn.CrossEntropyLoss()

    return None