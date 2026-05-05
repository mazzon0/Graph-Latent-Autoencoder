from .cnn_autoencoder import CnnAutoencoder

def get_model(name: str, config: dict):
    image_shape = config.get('image_shape', [3, 64, 64])
    channels = config.get('channels', [3, 8, 16, 32])
    mlp_sizes = config.get('mlp_sizes', [2048, 1024, 512])

    print("Model: ", end="")
    match(name):
        case 'cnn':
            print("cnn")
            return CnnAutoencoder(image_shape, channels, mlp_sizes)

    return None