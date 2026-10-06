from .graph_latent_autoencoder import GraphLatentAutoencoder


class SetLatentAutoencoder(GraphLatentAutoencoder):
    """
    An Autoencoder whose latent space is an unordered SET of nodes plus a global token, with no relations between
    the nodes. It is the middle arm of the experiment:

        1 vector                        CnnAutoencoder          the latent space is one flat embedding
        N nodes + global                SetLatentAutoencoder    the latent space is a set of embeddings
        N nodes + global + N^2 edges    GraphLatentAutoencoder  the latent space is a graph

    so that the gain of the graph over the set isolates the contribution of the EDGES, and the gain of the set
    over the vector isolates the contribution of splitting the latent space into several addressable slots.

    This is deliberately not a separate implementation: it is GraphLatentAutoencoder with use_edges=False, which
    removes the edge predictor, the edge attention bias and the edge update of the GNN, and the edge slots of the
    decoder. Everything else runs the exact same code, so a difference in the results can only come from the edges.
    A copy of the encoder would be easier to read, but the two arms would drift apart with every change, and a
    confounded comparison proves nothing.

    What is NOT changed on purpose: the DETR query layout stays [global, edge context, node 0, node 1, ...], so
    `num_queries: 64` gives 62 nodes in both arms and the transformer decoder sees the same number of queries.
    The edge context query is simply never read here. It costs `d_model` unused parameters, which buys an
    identical node count and an identical query set across the two arms: worth it for the comparison.

    Cost of the edges, measured on the default configuration (62 nodes, d_model 128, 4 GNN layers): the edge
    parameters (the predictor, the attention biases, the edge updates, the decoder edge slots) are 136,837 of
    2,711,874, so only 5% of the graph arm. The two arms are therefore close enough in capacity that the
    comparison is not badly confounded, but the counts should still be reported.
    The compute is a different story: the edge predictor runs N^2 = 3844 times per image instead of N = 62, and
    the latent holds 246,016 edge floats against 3,968 node floats. That is where the graph arm has to pay off.
    """
    def __init__(self, *args, **kwargs):
        assert not kwargs.pop('use_edges', False), "SetLatentAutoencoder is the arm without edges: use_edges cannot be True"
        super().__init__(*args, use_edges=False, **kwargs)
