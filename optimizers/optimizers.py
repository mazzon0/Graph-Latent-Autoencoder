import torch

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
MOMENTUM = 0.9

def get_optimizer(model: torch.nn.Module, name: str, config: dict):
    global LEARNING_RATE, WEIGHT_DECAY
    LEARNING_RATE = config.get('lr', LEARNING_RATE)
    WEIGHT_DECAY = config.get('weight_decay', WEIGHT_DECAY)
    MOMENTUM = config.get('momentum', MOMENTUM)

    match(name):
        case "adamw": return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        case "sgd":   return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    return None