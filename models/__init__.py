from .cnn_autoencoder import CnnAutoencoder
from .graph_latent_autoencoder import GraphLatentAutoencoder

def get_model(name: str, config: dict):
    image_shape = config.get('image_shape', [3, 64, 64])
    channels = config.get('channels', [3, 8, 16, 32])
    mlp_sizes = config.get('mlp_sizes', [2048, 1024, 512])
    d_model = config.get('d_model', 256)
    nhead = config.get('nhead', 8)
    num_encoder_layers = config.get('num_encoder_layers', 6)
    num_decoder_layers = config.get('num_decoder_layers', 6)
    dim_ff = config.get('dim_ff', 2048)
    dropout = config.get('dropout', 0.1)
    activation = config.get('activation', "relu")
    max_seq_len = config.get('max_seq_len', 5000)
    num_queries = config.get('num_queries', 64)
    d_node = config.get('d_node', 64)
    d_edge = config.get('d_edge', 64)
    d_global = config.get('d_global', 64)
    gnn_layers = config.get('gnn_layers', 4)
    train_with_sigmoid = config.get('train_with_sigmoid', True)

    print("Model: ", end="")
    match(name):
        case 'cnn':
            print("cnn")
            return CnnAutoencoder(image_shape, channels, mlp_sizes)
        case 'graph':
            print('graph')
            return GraphLatentAutoencoder(
                image_shape,
                channels,
                d_model,
                nhead,
                num_encoder_layers,
                num_decoder_layers,
                dim_ff,
                dropout,
                activation,
                max_seq_len,
                num_queries,
                d_node,
                d_edge,
                d_global,
                gnn_layers,
                train_with_sigmoid)

    return None