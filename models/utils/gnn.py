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

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, global_attr: torch.Tensor,
                node_keep: torch.Tensor = None, edge_keep: torch.Tensor = None):
        """
        Args:
            nodes (torch.Tensor): Shape (B, N, d_node) -> Note: Node 0 is your Global Token!
            edges (torch.Tensor): Shape (B, N, N, d_edge)
            node_keep (torch.Tensor, optional): Boolean mask (B, N, 1) of the nodes that are not pruned.
                Pruned nodes are ignored by the attention and their output is zero.
            edge_keep (torch.Tensor, optional): Boolean mask (B, N, N, 1) of the edges that are not pruned (their output is zero).
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
        if node_keep is not None:
            # pruned nodes cannot be attended (-1e9 instead of -inf, so that an image with no node left does not give NaN)
            total_scores = total_scores.masked_fill(~node_keep.transpose(1, 2), -1e9)
        attention_weights = torch.softmax(total_scores, dim=-1)
        
        node_context = torch.bmm(attention_weights, V)
        new_nodes = self.node_update(node_context)
        if node_keep is not None:
            new_nodes = new_nodes * node_keep
        
        # Update Edges based on the new node representations
        nodes_i = new_nodes.unsqueeze(2).expand(B, N, N, -1)
        nodes_j = new_nodes.unsqueeze(1).expand(B, N, N, -1)
        # (B, d_global) -> (B, 1, 1, d_global) -> (B, N, N, d_global)
        global_expanded = global_attr.unsqueeze(1).unsqueeze(2).expand(B, N, N, self.d_global)
        edge_inputs = torch.cat([edges, nodes_i, nodes_j, global_expanded], dim=-1)
        new_edges = self.edge_update(edge_inputs)
        if edge_keep is not None:
            new_edges = new_edges * edge_keep

        # TODO Update Global Embedding
        
        return new_nodes, new_edges, global_attr
