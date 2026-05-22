import torch
import torch.nn as nn
from .base_autoencoder import BaseAutoencoder

class CnnAutoencoder(BaseAutoencoder):
    """
    A Deep Convolutional Autoencoder with a fully-connected MLP bottleneck.
    """
    def __init__(self, image_shape: list, channels: list, mlp_sizes: list, train_with_sigmoid: bool = True):
        """
        Args:
            image_shape (list): [C, H, W] of the input image. 
                Constraints: H and W must be powers of 2.
            channels (list): List of filter counts for the CNN.
                N channels create N-1 CNN layers (3x3 kernel, stride=2).
                Example: [3, 16, 32, 64]. First element must match image_shape[0].
            mlp_sizes (list): List of hidden units for the MLP bottleneck.
                Example: [1024, 512, 256]. First element must match the 
                flattened output of the final CNN encoder layer.
            train_with_sigmoid (bool): Validation always applies sigmoid at the end.
                Training applies sigmoid depending on this value.
                Example: nn.BCEWithLogits loss needs the logits during the training phase,
                requiring this parameter set to False.
        """
        super().__init__()
        assert(image_shape[0] == channels[0])   # The CNN input channels do not meet the image channels
        assert((image_shape[1] & (image_shape[1] - 1)) == 0)    # Image height is not a power of 2
        assert((image_shape[2] & (image_shape[2] - 1)) == 0)    # Image width is not a power of 2
        self.image_shape = image_shape
        assert(channels[-1] * self.image_shape[1] // 2**(len(channels)-1) * self.image_shape[2] // 2**(len(channels)-1) == mlp_sizes[0])    # The CNN output size does not meet the MLP input size
        self.train_with_sigmoid = train_with_sigmoid

        # CNN Encoder
        self.cnn_encoder = nn.Sequential()
        for i in range(len(channels) - 1):
            self.cnn_encoder.add_module(f"enc_conv_{i}", nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1))
            if i != len(channels) - 2 or len(mlp_sizes) != 0:
                self.cnn_encoder.add_module(f"enc_conv_bn_{i}", nn.BatchNorm2d(channels[i+1]))
                self.cnn_encoder.add_module(f"enc_conv_act_{i}", nn.LeakyReLU())

        # MLP Encoder
        self.mlp_encoder = nn.Sequential(nn.Flatten())
        for i in range(len(mlp_sizes) - 1):
            self.mlp_encoder.add_module(f"enc_mlp_{i}", nn.Linear(in_features=mlp_sizes[i], out_features=mlp_sizes[i+1]))

            if i != len(mlp_sizes)-2:
                self.mlp_encoder.add_module(f"enc_mlp_bn_{i}", nn.BatchNorm1d(mlp_sizes[i+1]))
                self.mlp_encoder.add_module(f"enc_mlp_act_{i}", nn.LeakyReLU())

        # MLP Decoder
        self.mlp_decoder = nn.Sequential()
        for i in range(len(mlp_sizes)-1, 0, -1):
            self.mlp_decoder.add_module(f"dec_mlp_{i}", nn.Linear(in_features=mlp_sizes[i], out_features=mlp_sizes[i-1]))
            self.mlp_decoder.add_module(f"dec_mlp_bn_{i}", nn.BatchNorm1d(mlp_sizes[i-1]))
            self.mlp_decoder.add_module(f"dec_mlp_act_{i}", nn.LeakyReLU())
        self.mlp_decoder.add_module('unflatten',
            nn.Unflatten(1, [channels[-1], self.image_shape[1] // 2**(len(channels)-1), self.image_shape[2] // 2**(len(channels)-1)]))

        # CNN Decoder
        self.cnn_decoder = nn.Sequential()
        for i in range(len(channels)-1, 0, -1):
            self.cnn_decoder.add_module(f"dec_tconv_{i}",
                nn.ConvTranspose2d(channels[i], channels[i-1], kernel_size=3, stride=2, padding=1, output_padding=1))
            
            if i != 1:  # every layer has a Leaky ReLU activation, but the last one sigmoid activation for normalized values (0, 1)
                self.cnn_decoder.add_module(f"dec_tconv_bn_{i}", nn.BatchNorm2d(channels[i-1]))
                self.cnn_decoder.add_module(f"dec_tconv_act_{i}", nn.LeakyReLU())

        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights using Xavier uniform initialization.
        """
        for p in self.parameters():
            if isinstance(p, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(p.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
                
                if p.bias is not None:
                    nn.init.constant_(p.bias, 0)
                    
            elif isinstance(p, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(p.weight, 1)
                nn.init.constant_(p.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Passes the input through the full autoencoder pipeline.
        
        Args:
            x (torch.Tensor): Input image batch of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Reconstructed image batch of shape (N, C, H, W).
        """
        assert(self.image_shape[0] == x.shape[1])
        assert(self.image_shape[1] == x.shape[2])
        assert(self.image_shape[2] == x.shape[3])

        x = self.cnn_encoder(x)
        x = self.mlp_encoder(x)
        x = self.mlp_decoder(x)
        x = self.cnn_decoder(x)

        if not self.training or self.train_with_sigmoid:
            x = torch.sigmoid(x)

        return {
            'image': x,
            'nodes': torch.zeros(1, dtype=torch.float32, device=x.device),
            'edges': torch.zeros(1, dtype=torch.float32, device=x.device)
        }
    
    def get_first_layer(self):
        """Returns the first layer of the CNN encoder"""
        return self.cnn_encoder[0]