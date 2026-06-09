import torch
import torch.nn as nn
from .base_autoencoder import BaseAutoencoder
from .utils.transformer import DetrEmbeddingTransformer
from .utils.gnn import AttentionGraphBlock, GraphToImageDecoder

class GraphLatentAutoencoder(BaseAutoencoder):
    """
    An Autoencoder with a latent space with a graph structure.
    """
    def __init__(self,
                 image_shape: list,
                 channels: list,
                 d_model: int = 128, 
                 nhead: int = 8, 
                 num_encoder_layers: int = 6, 
                 num_decoder_layers: int = 6, 
                 dim_ff: int = 2048, 
                 dropout: float = 0.1, 
                 activation: str = "relu",
                 max_seq_len: int = 5000,
                 num_queries: int = 100,
                 d_node: int = 64,
                 d_edge: int = 64,
                 d_global: int = 64,
                 gnn_layers: int = 4,
                 train_with_sigmoid: bool = True):
        """
        Args:
            image_shape (list): [C, H, W] of the input image. 
                Constraints: H and W must be powers of 2.
        """
        super().__init__()
        assert(image_shape[0] == channels[0])   # The CNN input channels do not meet the image channels
        assert((image_shape[1] & (image_shape[1] - 1)) == 0)    # Image height is not a power of 2
        assert((image_shape[2] & (image_shape[2] - 1)) == 0)    # Image width is not a power of 2
        assert(channels[-1] == d_model)    # The CNN output size does not meet the MLP input size

        self.image_shape = image_shape
        self.d_model = d_model
        self.num_queries = num_queries
        self.d_node = d_node
        self.num_decoder_layers = num_decoder_layers

        # CNN Feature Extraction
        self.cnn_encoder = nn.Sequential()
        for i in range(len(channels) - 1):
            self.cnn_encoder.add_module(f"enc_conv_{i}", nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1))
            if i != len(channels) - 2:
                self.cnn_encoder.add_module(f"enc_conv_bn_{i}", nn.BatchNorm2d(channels[i+1]))
                self.cnn_encoder.add_module(f"enc_conv_act_{i}", nn.LeakyReLU())

        # Transformer
        self.transformer = DetrEmbeddingTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation=activation,
            max_seq_len=max_seq_len,
            num_queries=num_queries
        )

        # Nodes / Edges / Global tokens processing
        self.node_predictor = nn.Sequential(
            nn.Linear(d_model, d_node * 2),
            nn.BatchNorm1d(d_node * 2),
            nn.LeakyReLU(),
            nn.Linear(d_node * 2, d_node + 1)   # 'd_node' nodes + 1 confidence score
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_edge + 1)      # 'd_edge' edges + 1 confidence score
        )

        self.global_predictor = nn.Sequential(
            nn.Linear(d_model, d_global * 2),
            nn.BatchNorm1d(d_global * 2),
            nn.LeakyReLU(),
            nn.Linear(d_global * 2, d_global)
        )

        # GNN
        self.gnn = nn.ModuleList()
        for i in range(gnn_layers):
            self.gnn.append(AttentionGraphBlock(d_node, d_edge, d_global))

        # Image Generation
        self.image_generator = GraphToImageDecoder(d_node, init_channels=256, init_size=4, out_channels=3, train_with_sigmoid=train_with_sigmoid)

        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights using Xavier uniform initialization.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if hasattr(self, 'transformer') and hasattr(self.transformer, 'query_embed'):
            nn.init.xavier_uniform_(self.transformer.query_embed)

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

        # CNN Feature Extractor
        features = self.cnn_encoder(x)
        features_flattened = features.flatten(2).transpose(1, 2)

        # DETR Transformer
        tokens = self.transformer(features_flattened)   # Tokens: [Global, Edges, Node 0, Node 1, ...]
        global_token = tokens[:, 0, :]   # Shape: (B, d_model)
        edge_token = tokens[:, 1, :]     # Shape: (B, d_model)
        node_tokens = tokens[:, 2:, :]   # Shape: (B, N, d_model)

        # Nodes
        B, N, D = node_tokens.shape
        nodes_flat = node_tokens.reshape(B * N, D)
        nodes_out_flat = self.node_predictor(nodes_flat)
        nodes = nodes_out_flat.reshape(B, N, -1)

        # Edges
        nodes_i = node_tokens.unsqueeze(2).expand(B, N, N, D)   # (B, N, 1, D) -> (B, N, N, D)
        nodes_j = node_tokens.unsqueeze(1).expand(B, N, N, D)   # (B, 1, N, D) -> (B, N, N, D)
        edge_context = edge_token.unsqueeze(1).unsqueeze(2).expand(B, N, N, D)  # (B, 1, 1, D) -> (B, N, N, D)
        edge_inputs = torch.cat([edge_context, nodes_i, nodes_j], dim=-1)
        edges_flat = edge_inputs.reshape(B * N * N, D * 3)
        edges_out_flat = self.edge_predictor(edges_flat)
        edges = edges_out_flat.reshape(B, N, N, -1)

        # Global
        global_out = self.global_predictor(global_token)

        # Apply confidence scores
        node_features = nodes[..., :-1]               # (B, N, d_node)
        node_conf = torch.sigmoid(nodes[..., -1:])    # (B, N, 1)
        nodes_out = node_features * node_conf
        
        edge_features = edges[..., :-1]               # (B, N, N, d_edge)
        edge_conf = torch.sigmoid(edges[..., -1:])    # (B, N, N, 1)
        edges_out = edge_features * edge_conf

        node_features = nodes[..., :-1]               
        node_conf = torch.sigmoid(nodes[..., -1:])    # (B, N, 1)
        nodes_out = node_features * node_conf          
        
        edge_features = edges[..., :-1]               
        edge_conf = torch.sigmoid(edges[..., -1:])    # (B, N, N, 1)
        edges_out = edge_features * edge_conf

        # GNN
        gnn_nodes = nodes_out
        gnn_edges = edges_out
        gnn_global = global_out
        
        for gnn_layer in self.gnn:
            gnn_nodes, gnn_edges, gnn_global = gnn_layer(gnn_nodes, gnn_edges, gnn_global)

        # Image Generation
        reconstructed_image = self.image_generator(gnn_nodes)

        return {
            'image': reconstructed_image,
            'nodes': gnn_nodes,
            'edges': gnn_edges,
            'global': gnn_global,
            'node_conf': node_conf,
            'edge_conf': edge_conf
        }
    
    def get_first_layer(self):
        """Returns the first layer of the CNN encoder"""
        return self.cnn_encoder[0]

    @torch.no_grad()
    def export_for_inspector(self, x: torch.Tensor, batch_idx: int = 0) -> dict:
        """
        Runs a forward pass on an image tensor and completely structures the output dictionary
        to be perfectly compliant with the Streamlit Scene Graph UI specification.
        """
        # Gather raw computational variables
        out = self.forward(x)
        
        gnn_nodes_sample = out['nodes'][batch_idx]       # Shape: (N, d_node)
        edge_conf_sample = out['edge_conf'][batch_idx]   # Shape: (N, N, 1)
        node_conf_sample = out['node_conf'][batch_idx]   # Shape: (N, 1)
        
        num_nodes = gnn_nodes_sample.shape[0]
        num_layers = self.num_decoder_layers
        
        # Build Dense (N, N) Adjacency Matrix into Flattened Coordinate Lists
        # Generates row indices and col indices for all possible directed edges
        rows, cols = torch.meshgrid(torch.arange(num_nodes, device=x.device), 
                                    torch.arange(num_nodes, device=x.device), 
                                    indexing='ij')
        edge_indices = torch.stack([rows.flatten(), cols.flatten()], dim=-1)    # Shape: (N*N, 2)
        edge_conf_flat = edge_conf_sample.reshape(-1)                           # Shape: (N*N,)
        
        # Create default string structures and boolean masks for UI initialization
        node_conf_flat = node_conf_sample.squeeze(-1)
        default_node_names = [f"Object {i}" for i in range(num_nodes)]
        default_edge_names = [f"Rel {src.item()}➔{dst.item()}" for src, dst in edge_indices]
        
        # Default boolean visualization arrays (using a 0.5 confidence threshold)
        default_node_mask = node_conf_flat > 0.5
        default_edge_mask = edge_conf_flat > 0.5
        
        # Handle Attention Maps
        map_h, map_w = x.shape[2] // 8, x.shape[3] // 8
        
        dummy_node_attn = torch.rand(num_nodes, num_layers, map_h, map_w, device=x.device)
        dummy_global_attn = torch.rand(num_layers, map_h, map_w, device=x.device)
        dummy_relation_attn = torch.rand(num_layers, map_h, map_w, device=x.device)
        
        # Assemble final compliant package structure
        inspector_payload = {
            "metadata": {
                "num_nodes": num_nodes,
                "num_layers": num_layers
            },
            "nodes": {
                "names": default_node_names,
                "selected": default_node_mask.cpu(),
                "confidences": node_conf_flat.cpu(),
                "embeddings": gnn_nodes_sample.cpu(),
                "attention_maps": dummy_node_attn.cpu()
            },
            "edges": {
                "names": default_edge_names,
                "selected": default_edge_mask.cpu(),
                "confidences": edge_conf_flat.cpu(),
                "indices": edge_indices.cpu()
            },
            "global_tokens": {
                "global_attention": dummy_global_attn.cpu(),
                "relation_attention": dummy_relation_attn.cpu()
            }
        }
        
        return inspector_payload