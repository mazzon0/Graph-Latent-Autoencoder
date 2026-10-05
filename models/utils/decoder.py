"""
Decoders: graph -> image.

All the decoders share the same interface, so they are interchangeable:

    forward(nodes   (B, N, d_node),
            edges   (B, N, N, d_edge)  (or None),
            global_ (B, d_global))  ->  image logits (B, C, H, W)

The sigmoid, if needed, is applied outside of the model (see models.apply_sigmoid_if_requested).

Presence is read from the values, not from a separate mask: a pruned node or edge is a vector of exact zeros
(the gating multiplies the features by `node_keep` / `edge_keep`), and a token counts as present when
`vector.abs().sum() > 0`. This keeps the decoders decoupled from the gating mechanics, at the price of no
gradient flowing from the decoder back into the pruning decision. The global token is always present.

Decoders:
    'slot'      SlotBroadcastDecoder    MONet / Slot Attention style: every token is broadcast over a grid, a
                                        shared conv net turns it into a feature map plus a mask logit, and the
                                        masks are softmaxed over the tokens (each pixel is explained by the
                                        tokens that claim it). Absent tokens draw nothing.
    'crossattn' CrossAttentionDecoder   the grid positions are queries that cross-attend to the graph tokens.

`edge_slots` is the "does the decoder read the edges at all" knob: with 0 the decoder only sees the nodes and
the global token (the set arm of the ablation), with K > 0 the K strongest edges (by feature norm) become
tokens of their own, so the edges change what is rendered instead of only biasing the GNN aggregation.
"""
import math
import torch
import torch.nn as nn


def fourier_grid(res: int, n_freq: int) -> torch.Tensor:
    """
    Coordinates of a res x res grid in [-1, 1], as (x, y) followed by their sines and cosines at `n_freq`
    frequencies. Shape (res * res, 2 + 4 * n_freq), in row-major order (matching a (res, res) reshape).
    """
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, res), torch.linspace(-1, 1, res), indexing='ij')
    xy = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
    freqs = (2.0 ** torch.arange(n_freq)) * math.pi
    ang = xy.unsqueeze(-1) * freqs                                      # (P, 2, n_freq)
    return torch.cat([xy, ang.sin().flatten(1), ang.cos().flatten(1)], dim=-1)


class Upscaler(nn.Module):
    """
    (B, in_channels, res, res) -> (B, out_channels, size, size) logits, doubling the resolution at every stage.
    `size / res` must be a power of 2. The channels halve at every stage, starting from `base_channels` and
    never going below 32.
    """
    def __init__(self, in_channels: int, out_channels: int, res: int, size: int, base_channels: int = 128):
        super().__init__()
        assert size >= res and size % res == 0, f"the image size ({size}) must be a multiple of the decoder resolution ({res})"
        stages = int(math.log2(size // res))
        assert res * 2 ** stages == size, f"the image size ({size}) is not the decoder resolution ({res}) times a power of 2"

        layers = []
        channels = in_channels
        for i in range(stages):
            next_channels = max(base_channels >> i, 32)
            layers += [
                nn.ConvTranspose2d(channels, next_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(next_channels),
                nn.LeakyReLU()
            ]
            channels = next_channels
        layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphTokens(nn.Module):
    """
    Turns (nodes, edges, global) into a flat set of tokens (B, S, d) with a validity mask (B, S), so that the
    decoders below can treat the graph as a set of things to render.

    S = N nodes + 1 global + `edge_slots` edges. The edges kept are the `edge_slots` strongest ones of the
    batch element, ranked by feature norm (a pruned edge has norm 0, so it is picked last and marked invalid).
    Each token carries a learned type embedding, so the net can tell a node from an edge from the global token.
    """
    def __init__(self, d_node: int, d_edge: int, d_global: int, d: int, edge_slots: int = 0):
        super().__init__()
        self.edge_slots = edge_slots
        self.node_proj = nn.Linear(d_node, d)
        self.global_proj = nn.Linear(d_global, d)
        # an edge token also gets its two endpoints: an edge feature alone does not say what it connects
        self.edge_proj = nn.Linear(d_edge + 2 * d_node, d) if edge_slots > 0 else None
        self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)      # node / global / edge

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, global_: torch.Tensor):
        """
        Returns:
            tokens (B, S, d), valid (B, S) boolean.
        """
        B, N, d_node = nodes.shape

        tokens = [
            self.node_proj(nodes) + self.type_emb[0],                                    # (B, N, d)
            (self.global_proj(global_) + self.type_emb[1]).unsqueeze(1)                  # (B, 1, d)
        ]
        valid = [
            nodes.abs().sum(-1) > 0,                                                     # (B, N)
            torch.ones(B, 1, dtype=torch.bool, device=nodes.device)                      # the global is always present
        ]

        if self.edge_slots > 0:
            assert edges is not None, "the decoder was built with edge_slots > 0, but no edges were given"
            d_edge = edges.shape[-1]
            k = min(self.edge_slots, N * N)
            score = edges.norm(dim=-1).reshape(B, N * N)                                 # 0 for the pruned edges
            top, idx = score.topk(k, dim=1)                                              # (B, k)

            edges_flat = edges.reshape(B, N * N, d_edge)
            selected = edges_flat.gather(1, idx.unsqueeze(-1).expand(-1, -1, d_edge))    # (B, k, d_edge)
            source = nodes.gather(1, (idx // N).unsqueeze(-1).expand(-1, -1, d_node))    # (B, k, d_node)
            target = nodes.gather(1, (idx % N).unsqueeze(-1).expand(-1, -1, d_node))     # (B, k, d_node)

            tokens.append(self.edge_proj(torch.cat([selected, source, target], dim=-1)) + self.type_emb[2])
            valid.append(top > 0)

        return torch.cat(tokens, dim=1), torch.cat(valid, dim=1)


class SlotBroadcastDecoder(nn.Module):
    """
    Every token is broadcast over a res x res grid, summed with the encoded grid coordinates, and passed through
    a small conv net shared by all the tokens, which gives `feat` feature channels plus one mask logit. The masks
    are softmaxed over the tokens, so each pixel is split among the tokens that claim it, and the feature maps are
    summed with those weights. The composite is upscaled to the image.

    Absent tokens get a mask logit of -inf, so they draw nothing and receive no gradient.

    `layers` is the depth of the per token net and `upscaler_channels` the width of the first upscaling stage.

    Cost warning: the conv net runs on B * S maps, so the compute and the activation memory grow linearly with the
    number of nodes AND with the batch size. Slot Attention usually has under a dozen slots; a DETR encoder has
    `num_queries - 2`. `res`, `hidden`, `feat` and `layers` are the knobs to trade this off; the upscaler runs on
    the composite only (B maps), so `upscaler_channels` is cheap capacity.
    """
    def __init__(self, d_node: int, d_edge: int, d_global: int, image_shape: list,
                 res: int = 16, hidden: int = 64, feat: int = 32, n_freq: int = 4, edge_slots: int = 0,
                 layers: int = 3, upscaler_channels: int = 128):
        super().__init__()
        self.res = res
        self.feat = feat
        self.tokens = GraphTokens(d_node, d_edge, d_global, hidden, edge_slots)
        self.register_buffer('coords', fourier_grid(res, n_freq))
        self.pos_proj = nn.Linear(self.coords.shape[-1], hidden)
        # no normalization here: the batch of maps is full of absent tokens, whose statistics are meaningless
        assert layers >= 1, "the per token net needs at least the output layer"
        net = []
        for _ in range(layers - 1):
            net += [nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1), nn.LeakyReLU()]
        net.append(nn.Conv2d(hidden, feat + 1, kernel_size=3, stride=1, padding=1))  # feat channels + 1 mask logit
        self.net = nn.Sequential(*net)
        self.upscaler = Upscaler(feat, image_shape[0], res, image_shape[1], base_channels=upscaler_channels)
        self.last_masks = None      # (B, S, res, res) of the last forward pass, for inspection

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor = None, global_: torch.Tensor = None):
        tokens, valid = self.tokens(nodes, edges, global_)                          # (B, S, hidden), (B, S)
        B, S, hidden = tokens.shape

        broadcast = tokens.unsqueeze(2) + self.pos_proj(self.coords).unsqueeze(0).unsqueeze(0)   # (B, S, P, hidden)
        broadcast = broadcast.reshape(B * S, self.res, self.res, hidden).permute(0, 3, 1, 2)

        out = self.net(broadcast).reshape(B, S, self.feat + 1, self.res, self.res)
        features, mask = out[:, :, :-1], out[:, :, -1]                             # (B, S, feat, res, res), (B, S, res, res)

        mask = mask.masked_fill(~valid.unsqueeze(-1).unsqueeze(-1), torch.finfo(mask.dtype).min)
        weights = torch.softmax(mask, dim=1)                                       # each pixel sums to 1 over the tokens
        self.last_masks = weights.detach()

        composite = (weights.unsqueeze(2) * features).sum(dim=1)                   # (B, feat, res, res)
        return self.upscaler(composite)


class CrossAttentionDecoder(nn.Module):
    """
    The queries are the encoded positions of a res x res grid. A few transformer decoder layers let every position
    cross-attend to the graph tokens and self-attend to the other positions, so a pixel can gather from several
    nodes at once instead of being assigned to one. The absent tokens are excluded with a key padding mask.

    Cheaper than the slot decoder (the tokens are the attention memory, not a batch dimension), but it gives no
    per-token masks, so there is nothing to score against ground truth segmentations.
    """
    def __init__(self, d_node: int, d_edge: int, d_global: int, image_shape: list,
                 res: int = 16, d: int = 128, nhead: int = 4, layers: int = 3, n_freq: int = 4, edge_slots: int = 0):
        super().__init__()
        self.res = res
        self.d = d
        self.tokens = GraphTokens(d_node, d_edge, d_global, d, edge_slots)
        self.register_buffer('coords', fourier_grid(res, n_freq))
        self.pos_proj = nn.Linear(self.coords.shape[-1], d)
        layer = nn.TransformerDecoderLayer(d, nhead, dim_feedforward=2 * d, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.upscaler = Upscaler(d, image_shape[0], res, image_shape[1])

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor = None, global_: torch.Tensor = None):
        tokens, valid = self.tokens(nodes, edges, global_)                          # (B, S, d), (B, S)
        B = tokens.shape[0]

        queries = self.pos_proj(self.coords).unsqueeze(0).expand(B, -1, -1)         # (B, P, d)
        out = self.decoder(queries, tokens, memory_key_padding_mask=~valid)         # (B, P, d)

        return self.upscaler(out.transpose(1, 2).reshape(B, self.d, self.res, self.res))


def get_decoder(name: str, d_node: int, d_edge: int, d_global: int, image_shape: list, config: dict):
    """
    Builds the decoder named `name`, reading its own hyperparameters from `config`
    (the `decoder_<name>` section of the model config).
    """
    print("Decoder: ", end="")
    match name:
        case 'slot':
            print("slot")
            return SlotBroadcastDecoder(
                d_node, d_edge, d_global,
                image_shape,
                res=config.get('res', 16),
                hidden=config.get('hidden', 64),
                feat=config.get('feat', 32),
                n_freq=config.get('n_freq', 4),
                edge_slots=config.get('edge_slots', 0),
                layers=config.get('layers', 3),
                upscaler_channels=config.get('upscaler_channels', 128))
        case 'crossattn':
            print("crossattn")
            return CrossAttentionDecoder(
                d_node, d_edge, d_global,
                image_shape,
                res=config.get('res', 16),
                d=config.get('d', 128),
                nhead=config.get('nhead', 4),
                layers=config.get('layers', 3),
                n_freq=config.get('n_freq', 4),
                edge_slots=config.get('edge_slots', 0))
        case _:
            raise ValueError(f"unknown decoder '{name}' (expected 'slot' or 'crossattn')")
