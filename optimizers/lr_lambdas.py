import math

LEARNING_RATE = 1e-4
WARMUP_EPOCHS = 10
DECAY_RATE = 0.1
END_EPOCH = 49

def get_lr_lambda(name: str, config: dict, end_epoch: int):
    global LEARNING_RATE, WARMUP_EPOCHS, DECAY_RATE, START_EPOCH, END_EPOCH
    LEARNING_RATE = config.get('lr', LEARNING_RATE)
    WARMUP_EPOCHS = config.get('warmup_epochs', WARMUP_EPOCHS)
    DECAY_RATE = config.get('decay_rate', DECAY_RATE)

    END_EPOCH = end_epoch if end_epoch >= 0 else 0

    print("LR Scheduler: ", end="")
    match(name):
        case "constant":
            print("constant")
            return lr_lambda_constant
        case "exponential":
            print("exponential")
            return lr_lambda_exponential
        case "cosine":
            print("cosine")
            return lr_lambda_cosine
        case "cosine_with_warmup":
            print("cosine_with_warmup")
            return lr_lambda_cosine_with_warmup

def lr_lambda_constant(epoch):
    return 1.0

def lr_lambda_exponential(epoch):
    return (1.0 - DECAY_RATE) ** epoch

def lr_lambda_cosine(epoch):
    progress = float(epoch) / float(END_EPOCH)
    return 0.5 * (1.0 + math.cos(math.pi * progress))   # The '+1' is not correct, but it is added to not waste the last epoch with lr=0

def lr_lambda_cosine_with_warmup(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / WARMUP_EPOCHS
        else:
            progress = float(epoch - WARMUP_EPOCHS) / float(END_EPOCH - WARMUP_EPOCHS + 1)  # The '+1' is not correct, but it is added to not waste the last epoch with lr=0
            return 0.5 * (1.0 + math.cos(math.pi * progress))