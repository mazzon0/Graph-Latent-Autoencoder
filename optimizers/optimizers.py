import torch
import torch.nn as nn

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
MOMENTUM = 0.9

def parameter_groups(model: torch.nn.Module, weight_decay: float):
    """
    Splits the parameters in two groups: the ones that get the weight decay (the weights of the convolutions,
    of the linear layers and the embeddings) and the ones that must not (every bias, and the scales of the
    normalization layers). Decaying a LayerNorm scale or a bias shrinks it towards 0 for no good reason, which
    on a deep transformer costs accuracy.
    """
    if weight_decay == 0.0:
        return model.parameters()

    no_decay_modules = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.Embedding)
    no_decay = set()
    for module_name, module in model.named_modules():
        if isinstance(module, no_decay_modules):
            no_decay.update(f"{module_name}.{n}" if module_name else n for n, _ in module.named_parameters())

    decayed, plain = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (plain if name in no_decay or name.endswith('.bias') or parameter.dim() <= 1 else decayed).append(parameter)

    print(f"weight decay {weight_decay} on {sum(p.numel() for p in decayed):,} parameters, "
          f"none on {sum(p.numel() for p in plain):,} (biases and normalization scales) | ", end="")
    return [{'params': decayed, 'weight_decay': weight_decay},
            {'params': plain, 'weight_decay': 0.0}]

def get_optimizer(model: torch.nn.Module, name: str, config: dict):
    global LEARNING_RATE, WEIGHT_DECAY, MOMENTUM
    LEARNING_RATE = float(config.get('lr', LEARNING_RATE))
    WEIGHT_DECAY = float(config.get('weight_decay', WEIGHT_DECAY))
    MOMENTUM = float(config.get('momentum', MOMENTUM))

    print("Optimizer: ", end="")
    groups = parameter_groups(model, WEIGHT_DECAY)
    match(name):
        case "adamw":
            print("adamw")
            return torch.optim.AdamW(groups, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        case "sgd":
            print("sgd")
            return torch.optim.SGD(groups, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    return None