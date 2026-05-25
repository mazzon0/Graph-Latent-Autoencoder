import torch
import torch.nn as nn

class AttentionGraphBlock(nn.Module):
    """
    A Graph Network block that uses attention to update nodes 
    and global states, using edge features as an attention bias.
    """
    def __init__(self, d_node: int, d_edge: int, d_global: int):
        super().__init__()
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_global = d_global

        self.global_to_film = nn.Linear(d_global, d_node * 2)
        
        self.q_proj = nn.Linear(d_node, d_node)
        self.k_proj = nn.Linear(d_node, d_node)
        self.v_proj = nn.Linear(d_node, d_node)
        
        self.edge_to_bias = nn.Linear(d_edge, 1)    # for multi-head attention, change output to 8
        
        # MLPs for updating states post-attention
        self.node_update = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.LayerNorm(d_node),
            nn.LeakyReLU()
        )
        self.edge_update = nn.Sequential(
            nn.Linear(d_edge + d_node * 2 + d_global, d_edge),
            nn.LayerNorm(d_edge),
            nn.LeakyReLU()
        )

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, global_attr: torch.Tensor):
        """
        Args:
            nodes (torch.Tensor): Shape (B, N, d_node) -> Note: Node 0 is your Global Token!
            edges (torch.Tensor): Shape (B, N, N, d_edge)
        """
        B, N, _ = nodes.shape

        # Global token predicts a scale (gamma) and shift (beta) for the nodes
        film_params = self.global_to_film(global_attr).unsqueeze(1) # (B, 1, d_node * 2)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        modulated_nodes = nodes * (1 + gamma) + beta

        Q = self.q_proj(modulated_nodes) # (B, N, d_node)
        K = self.k_proj(modulated_nodes) # (B, N, d_node)
        V = self.v_proj(modulated_nodes) # (B, N, d_node)
        
        # (B, N, d_node) x (B, d_node, N) -> (B, N, N)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_node ** 0.5)
        
        # (B, N, N, d_edge) -> (B, N, N, 1) -> (B, N, N)
        edge_bias = self.edge_to_bias(edges).squeeze(-1)
        
        total_scores = attention_scores + edge_bias
        attention_weights = torch.softmax(total_scores, dim=-1)
        
        node_context = torch.bmm(attention_weights, V)
        new_nodes = self.node_update(node_context)
        
        # Update Edges based on the new node representations
        nodes_i = new_nodes.unsqueeze(2).expand(B, N, N, -1)
        nodes_j = new_nodes.unsqueeze(1).expand(B, N, N, -1)
        # (B, d_global) -> (B, 1, 1, d_global) -> (B, N, N, d_global)
        global_expanded = global_attr.unsqueeze(1).unsqueeze(2).expand(B, N, N, self.d_global)
        edge_inputs = torch.cat([edges, nodes_i, nodes_j, global_expanded], dim=-1)
        new_edges = self.edge_update(edge_inputs)

        # TODO Update Global Embedding
        
        return new_nodes, new_edges, global_attr
    

class GraphToPixelDecoder(nn.Module):
    """
    Inner Model: Generates exactly 1 latent pixel feature array 
    given the pooled graph context and a positional embedding.
    """
    def __init__(self, d_node: int, d_pos: int, init_channels: int):
        super().__init__()
        # Input features = Mean Node (d_node) + Max Node (d_node) + Pos Embedding (d_pos)
        input_dim = (d_node * 2) + d_pos
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            
            # Outputs the n-dimensional pixel feature array
            nn.Linear(512, init_channels),
            nn.LeakyReLU()
        )

    def forward(self, pooled_graph: torch.Tensor, pos_emb: torch.Tensor):
        """
        Args:
            pooled_graph (torch.Tensor): Shape (B, ..., d_node * 2)
            pos_emb (torch.Tensor): Shape (B, ..., d_pos)
        Returns:
            torch.Tensor: Shape (B, ..., init_channels) -> 1 Latent Pixel array
        """
        # Concatenate graph context and position information along the last dimension
        x = torch.cat([pooled_graph, pos_emb], dim=-1)
        return self.mlp(x)


'''class GraphToImageDecoder(nn.Module):
    """
    Decodes a graph into an image by pooling node features, passing them 
    through an MLP to form a low-res grid, and upscaling via CNN.
    """
    def __init__(self, d_node: int, init_channels: int = 256, init_size: int = 4, out_channels: int = 3, train_with_sigmoid: bool = True):
        super().__init__()
        self.init_channels = init_channels
        self.init_size = init_size
        self.train_with_sigmoid = train_with_sigmoid
        
        self.mlp_readout = nn.Sequential(
            nn.Linear(d_node * 2, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, init_channels * init_size * init_size),
            nn.LeakyReLU()
        )
        
        self.upscaler = nn.Sequential(
            # (Batch, 256, 4, 4) -> (Batch, 128, 8, 8)
            nn.ConvTranspose2d(init_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            # (Batch, 128, 8, 8) -> (Batch, 64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # (Batch, 64, 16, 16) -> (Batch, 32, 32, 32)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # (Batch, 32, 32, 32) -> (Batch, 3, 64, 64)
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, nodes: torch.Tensor):
        """
        Args:
            nodes (torch.Tensor): Shape (B, N, d_node)
        """
        B, N, D = nodes.shape
        
        # Graph Pooling
        mean_pool = nodes.mean(dim=1)           # Shape: (B, D)
        max_pool = nodes.max(dim=1).values      # Shape: (B, D)
        pooled_graph = torch.cat([mean_pool, max_pool], dim=-1)  # (B, D * 2)
        
        # MLP: low resolution image
        flat_grid = self.mlp_readout(pooled_graph)
        low_res_image = flat_grid.view(B, self.init_channels, self.init_size, self.init_size)
        
        # CNN Upscaler
        final_image = self.upscaler(low_res_image)

        if not self.training or self.train_with_sigmoid:
            final_image = torch.sigmoid(final_image)
        
        return final_image'''
    

class GraphToImageDecoder(nn.Module):
    """
    Wrapper Model: Stores positional embeddings, coordinates the Inner Model 
    to generate a low-res latent grid, and upscales it to a final image.
    """
    def __init__(self, d_node: int, d_pos: int = 32, init_channels: int = 256, 
                 init_size: int = 4, out_channels: int = 3, train_with_sigmoid: bool = True):
        super().__init__()
        self.init_channels = init_channels
        self.init_size = init_size
        self.train_with_sigmoid = train_with_sigmoid
        
        # Positional Embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(init_size, init_size, d_pos))
        
        # Inner Model
        self.inner_model = GraphToPixelDecoder(d_node, d_pos, init_channels)
        
        # CNN Upscaler
        self.upscaler = nn.Sequential(
            # (Batch, 256, 4, 4) -> (Batch, 128, 8, 8)
            nn.ConvTranspose2d(init_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            # (Batch, 128, 8, 8) -> (Batch, 64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # (Batch, 64, 16, 16) -> (Batch, 32, 32, 32)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # (Batch, 32, 32, 32) -> (Batch, 3, 64, 64)
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, nodes: torch.Tensor):
        """
        Args:
            nodes (torch.Tensor): Shape (B, N, d_node)
        """
        B, N, D = nodes.shape
        H_init, W_init = self.init_size, self.init_size
        
        mean_pool = nodes.mean(dim=1)           # (B, D)
        max_pool = nodes.max(dim=1).values      # (B, D)
        pooled_graph = torch.cat([mean_pool, max_pool], dim=-1)  # (B, D * 2)
        graph_context = pooled_graph.unsqueeze(1).unsqueeze(2).expand(-1, H_init, W_init, -1)
        pos_emb_expanded = self.pos_embeddings.unsqueeze(0).expand(B, -1, -1, -1)
        
        # (B, H_init, W_init, init_channels)
        latent_grid = self.inner_model(graph_context, pos_emb_expanded)
        
        # (B, 4, 4, 256) -> (B, 256, 4, 4)
        low_res_image = latent_grid.permute(0, 3, 1, 2)
        
        # Standard upscaling pass
        final_image = self.upscaler(low_res_image)

        if not self.training or self.train_with_sigmoid:
            final_image = torch.sigmoid(final_image)
        
        return final_image