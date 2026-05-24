import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the 
    embeddings in the sequence. For image data, this is applied to the 
    flattened spatial dimensions (H * W).
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        if self.batch_first:
            pe = pe.transpose(0, 1)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Flattened image embedding sequence.
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)


class DetrEmbeddingTransformer(nn.Module):
    """
    A Transformer adapted for parallel object detection (DETR-style).
    It uses learned object queries instead of autoregressive target sequences.
    """
    def __init__(self, 
                 d_model: int = 512, 
                 nhead: int = 8, 
                 num_encoder_layers: int = 6, 
                 num_decoder_layers: int = 6, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1, 
                 activation: str = "relu",
                 max_seq_len: int = 5000,
                 num_queries: int = 100):
        """
        Args:
            ... [Standard Transformer Args] ...
            num_queries (int): The maximum number of objects the model can detect per image.
        """
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.d_model = d_model
        self.num_queries = num_queries

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len, batch_first=True)
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )

    def forward(self, 
                src: torch.Tensor, 
                src_mask: torch.Tensor = None, 
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Passes the image features and learned queries through the Transformer.

        Args:
            src (torch.Tensor): The flattened image features from the CNN backbone.
                Shape: (Batch, H*W, d_model).
            src_mask (torch.Tensor, optional): Additive mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Boolean mask for padded image regions.

        Returns:
            torch.Tensor: The output predictions for the queries. Shape: (Batch, num_queries, d_model).
        """
        assert self.d_model == src.shape[-1], f"Source embedding dim {src.shape[-1]} does not match d_model {self.d_model}"
        batch_size = src.shape[0]

        src = self.pos_encoder(src)
        tgt = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)

        output = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=None,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=src_key_padding_mask
        )

        return output