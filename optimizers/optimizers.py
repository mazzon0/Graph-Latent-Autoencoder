import torch

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
MOMENTUM = 0.9

def get_optimizer(model: torch.nn.Module, name: str, config: dict):
    global LEARNING_RATE, WEIGHT_DECAY, MOMENTUM
    LEARNING_RATE = float(config.get('lr', LEARNING_RATE))
    WEIGHT_DECAY = float(config.get('weight_decay', WEIGHT_DECAY))
    MOMENTUM = float(config.get('momentum', MOMENTUM))

    print("Optimizer: ", end="")
    match(name):
        case "adamw":
            print("adamw")
            return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        case "sgd":
            print("sgd")
            return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    return None