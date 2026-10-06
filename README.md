# Graph Latent Autoencoder
This project aims to train self-supervised Scene Graph Generation models and graph-conditioned image generation models.
This is achieved with an image-to-graph-to-image autoencoder.

## Table of Contents
* [Architecture](#architecture)
* [Quick Start](#quick-start)
    * [Setup](#setup)
    * [Training](#training)
    * [Inference](#inference)
    * [Inspect](#inspect)
* [Configuration](#configuration)
    * [General](#general)
    * [Models](#models)
    * [Optimizers](#optimizers)
    * [Losses](#losses)
    * [Inference Configuration](#inference-configuration)

## Architecture

## Quick Start

### Setup

Download the repository.
```bash
git clone git@github.com:mazzon0/Graph-Latent-Autoencoder.git
cd Graph-Latent-Autoencoder
```

Create the virtual environment.
For Linux systems:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate
```
For Windows systems:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
Get-Content requirements.txt | Where-Object { $_ -notmatch '^(nvidia-|cuda-|triton|torch==|torchvision==)' } | Set-Content requirements-windows.txt
pip install -r requirements-windows.txt
```

Download the datasets: `coco` (default), `clevr` or `all`.
For Linux systems:
```bash
./download.sh clevr
```
For Windows systems:
```powershell
powershell -ExecutionPolicy Bypass -File download.ps1 clevr
```
CLEVR v1.0 is a 19 GB download, of which only the train/val images and the scene annotations (object attributes and relations) are extracted, in `data/datasets/clevr/CLEVR_v1.0`.
Select the dataset with the `dataset` field of the configuration file (`"coco"` or `"clevr"`).

**Dataset cache.** With `cache_dataset: true` (the default) the first run resizes every image once to the size given by `image_shape` in the model config, and stores the result next to the images (about 1.4 GB for COCO, 0.9 GB for CLEVR).
Later runs load that file instead, which is much faster. It is rebuilt when the number of images or the image size change, and deleting it forces a rebuild.
The augmentations are applied on the fly to the stored images, so crops are upscaled from the low resolution version and look softer than the validation images.
With `cache_dataset: false` every image is loaded and resized on demand: the same images, but slower. Use it when data loading is not the bottleneck, or when the dataset does not fit in memory.

### Training

Train the model (storing last and best model each epoch in its own experiment directory, see [Experiments](#experiments)). The model, the loss, the optimizer, the lr scheduler and other parameters can be customized on the configuration file.
Before training, you need to activate the virtual environment with `source .venv/bin/activate`.
```bash
python3 train.py configs/some_configuration.yaml
```
Then, you can deactivate the virtual environment with `deactivate`.

### Inference

Test the model on a single image. It will tell you the loss and the will save the original and reconstructed image side to side in `<experiment dir>/inference/<image name>.png`. You can use the same configuration file used for the training, specifying the model to use in the `from_checkpoint` field.
Before executing the model, you need to activate the virtual environment with `source .venv/bin/activate`.
```bash
python3 inference.py configs/some_configuration.yaml some_image.jpg
```
Then, you can deactivate the virtual environment with `deactivate`.

### Inspect

The inference script stores some data about the latent graph in `<experiment dir>/inference/<image name>.pt`.
You can open the inspector to analyze this file in the browser.
```bash
streamlit run inspector.py experiments/some_experiment/inference/some_image.pt
```

### Experiments

Each training run has its own directory in `experiments/`:
```
experiments/<name>/
    config.yaml      copy of the configuration used
    best.pth         checkpoint with the best validation loss
    last.pth         checkpoint of the last epoch
    metrics.jsonl    one json line per epoch: epoch, seconds, lr, train and val losses
    inference/       outputs of inference.py (<image name>.png and <image name>.pt)
```
The name is the optional `experiment_name` field of the configuration, `<model>_<timestamp>` by default.
Starting a run with the name of an existing experiment is refused, so results are never overwritten.
To resume a training, set `from_checkpoint: true` with the `experiment_name` of the existing experiment: the same directory is reused, the metrics are appended and a timestamped copy of the new configuration is saved next to the original.
`checkpoint_file` is looked up as given, and otherwise inside the experiment directory (`best.pth` refers to `experiments/<experiment_name>/best.pth`).
To start a new experiment from an old checkpoint, give an `experiment_name` and a path to the checkpoint (`checkpoint_file: "experiments/old/best.pth"`).

## Configuration

### General
All configuration files needs some general values.

`model` allows to select an autoencoder model (`cnn`, `set` or `graph`, see [Models](#models)), `optimizer` allows to select an optimizer and `dataset` allows to select a dataset.
`start_epoch` and `end_epoch` allows to choose the range of epochs for the training process. Epochs are 0 based. `start_epoch` and `end_epoch` are included in the range.

`experiment_name` (optional) is the name of the directory of the experiment.

It is possible to initialize the model and the optimizer from a checkpoint, by setting `from_checkpoint` to `true` and specifying the path of the model in `checkpoint_file`. If `from_checkpoint` is `false`, then `checkpoint_file` is ignored.

It is possible to set the `batch_size` and `num_workers` values, and to enable or disable the dataset cache with `cache_dataset` (see [Setup](#setup)).

About the loss, it is possible to select which `reconstruction_loss` to use (`hybrid` is a linear combination of `l1` and `ssim`). It is possible to select the regularizers for the latent graph: the sum of the probabilities `probs`, the count of elements `discr` or the `band` regularizer (see [Losses](#losses)). The CNN Autoencoder will return 0 for these values. It is possible to select the weights of these losses `alpha`, `beta` and `gamma` respectively for reconstruction loss, nodes regularizer and edge regularizer.

```yaml
model: "cnn"
optimizer: "adamw" | "sgd"
dataset: "coco"
start_epoch: 0
end_epoch: 59
from_checkpoint: false
checkpoint_file: "best.pth"
batch_size: 128
num_workers: 8
cache_dataset: true

loss:
  reconstruction: "l1" | "l2" | "ssim" | "bce" | "hybrid"
  nodes: "probs" | "discr"
  edges: "probs" | "discr"
  alpha: 1.0
  beta: 0.0
  gamma: 0.0
```

### Models

The **CNN Autoencoder** can be customized adding the field `model_cnn`. The model is composed of
CNN Encoder -> MLP Encoder -> MLP Decoder -> CNN Decoder. The encoder and the decoder are symmetrical.

`image_shape` is the resolution of the images, represented as a list `[channels, height, width]`.

`channels` allows to specify the number of channels for each intermediate representation of the CNN Encoder (and also CNN Decoder).
The number of layers of the CNN Encoder (and also CNN Decoder) is going to be `len(channels) - 1`.

`mlp_sizes` allows to select the size of the intermediate representation of the MLP Encoder (and also MLP Decoder).
The number of layers of the MLP Encoder (and also the MLP Decoder) is going to be `len(mlp_sizes) - 1`.

The output image is made of logits: the model does not apply any final activation.
`apply_sigmoid` decides whether a sigmoid is applied to them (mapping the values to [0, 1]) before computing the loss, in training, validation and inference.
It should be `true` for the `l1`, `l2`, `ssim` and `hybrid` losses, and `false` for `bce` (which works directly on the logits).
The images saved by the inference script always get the sigmoid.

```yaml
model_cnn:
  image_shape: [3, 64, 64]
  channels: [3, 64, 128, 256]
  mlp_sizes: [16384, 2048, 1024]
  apply_sigmoid: true
```

The **Graph Latent Autoencoder** can be customized adding the field `model_graph`.
The model is composed of CNN -> DETR -> Graph Generation -> GNN -> CNN Upscaler.

`image_shape` is the resolution of the images, represented as a list `[channels, height, width]`.

`channels` allows to specify the number of channels for each intermediate representation of the CNN
The number of layers of the CNN Encoder (and also CNN Decoder) is going to be `len(channels) - 1`.

`d_model` is the dimensionality of the input and output representations (embedding dimension) within the Transformer layers.

`nhead` is the number of attention heads in the multi-head attention mechanisms.

`num_encoder_layers` is the number of transformer encoder layers.

`num_decoder_layers` is the number of transformer decoder layers.

`dim_ff` is the dimensionality of the feed-forward network (FFN) models within the transformer layers.

`dropout` is the dropout probability applied to the transformer layers to prevent overfitting.

`activation` is the activation function used in the feed-forward network layers (e.g., "relu", "gelu").

`max_seq_len` is the maximum sequence length (or positional embedding limit) that the transformer can process.

`norm_first` puts the LayerNorm before the attention and the feed forward (pre-LN) instead of after (post-LN, the PyTorch default).
Pre-LN keeps the residual path clean, so a wide transformer trains without a carefully tuned warmup and tolerates a larger learning rate. It should be the same in both arms of a comparison.

`num_queries` is the number of object queries (learned positional embeddings) fed into the DETR decoder.
The first two queries are reserved for the global token and the edge context token, so the graph has `num_queries - 2` nodes
(`num_queries: 64` gives 62 nodes).

`d_node` is the feature dimension size for each node in the generated graph structure.

`d_edge` is the feature dimension size for each edge in the generated graph structure.

`d_global` is the feature dimension size for the global graph context vector (representing the graph properties as a whole).

`gnn_layers` is the number of message-passing layer iterations within the Graph Neural Network (GNN) module.

`node_threshold` and `edge_threshold` are the target thresholds of the gating (`0` means no pruning, the graph is dense).
Nodes with a confidence below `node_threshold` are pruned. Edges with a confidence below `edge_threshold` are pruned, and so are the edges of a pruned node.
Pruned nodes and edges are ignored by the GNN (they cannot be attended) and are zero vectors after it.
The thresholds start at 0 (dense graph) and grow linearly to their target, following the sparsity schedule (`delay_epochs` and `ramp_epochs` of the `loss` configuration, see [Losses](#losses)).

`decoder` chooses the graph to image decoder, and the `decoder_<name>` section holds its hyperparameters.

- `slot` broadcasts every token over a `res` x `res` grid and renders it with a `layers` deep conv net of width `hidden`, shared by all the tokens, into `feat` feature channels plus a mask logit. The masks are softmaxed over the tokens, so every pixel is split among the tokens that claim it, and the composite is upscaled to the image through stages starting at `upscaler_channels` channels. Its compute and its memory grow with the number of nodes times the batch size, while `upscaler_channels` runs on the composite only and is therefore cheap capacity.
- `crossattn` uses the positions of a `res` x `res` grid as queries that cross-attend to the tokens, through `layers` transformer decoder layers of width `d`.

`edge_slots` (for `slot` and `crossattn`) is the number of edges the decoder renders as tokens of their own: the strongest ones by feature norm, each one carrying its two endpoints.
With `edge_slots: 0` the decoder only sees the nodes and the global token, which is the ablation arm where the latent space is a set rather than a graph.
`n_freq` is the number of frequencies of the sinusoidal encoding of the grid coordinates.

A pruned node or edge is a zero vector, and the decoders read it as absent: it gets no mask and draws nothing. The global token is always present.

The output image is made of logits: the model does not apply any final activation.
`apply_sigmoid` decides whether a sigmoid is applied to them (mapping the values to [0, 1]) before computing the loss, in training, validation and inference.
It should be `true` for the `l1`, `l2`, `ssim` and `hybrid` losses, and `false` for `bce` (which works directly on the logits).
The images saved by the inference script always get the sigmoid.

```yaml
model_graph:
  image_shape: [3, 64, 64]
  channels: [3, 64, 192, 512]
  d_model: 512
  nhead: 8
  num_encoder_layers: 6
  num_decoder_layers: 6
  dim_ff: 2048
  dropout: 0.1
  activation: "relu"
  norm_first: true
  max_seq_len: 5000
  num_queries: 32
  d_node: 64
  d_edge: 64
  d_global: 64
  gnn_layers: 6
  node_threshold: 0.0
  edge_threshold: 0.0
  decoder: "slot"
  decoder_slot:
    res: 16
    hidden: 256
    feat: 128
    n_freq: 4
    layers: 3
    upscaler_channels: 512
    edge_slots: 8
  decoder_crossattn:
    res: 16
    d: 128
    nhead: 4
    layers: 3
    n_freq: 4
    edge_slots: 8
  apply_sigmoid: true
```

This is the ~50M parameter configuration of [configs/graph.yaml](configs/graph.yaml); `configs/set.yaml` is the same with
`edge_slots: 0`. The memory of the slot decoder is `batch x slots x res^2 x hidden`, so it, and not the transformer,
decides the batch size: on an 8 GiB card the set arm fits a batch of 48 and the graph arm a batch of 32.

The **Set Latent Autoencoder** (`model: "set"`, field `model_set`) is the same model with the edges removed:
its latent space is an unordered set of nodes plus the global token. It takes exactly the same fields as the graph
model, and `d_edge`, `edge_threshold` and the `edges` regularizer are simply ignored (`edge_slots` must be `0`).

It is the middle arm of the experiment:

| model | latent space | what it adds |
| --- | --- | --- |
| `cnn` | one vector | - |
| `set` | `num_queries - 2` nodes + a global token | several addressable slots |
| `graph` | the same nodes + a global token + `(num_queries - 2)^2` edges | relations between the slots |

so the gap between `graph` and `set` measures what the edges are worth, and the gap between `set` and `cnn` measures
what splitting the latent space into slots is worth.

It is not a separate implementation: it is `GraphLatentAutoencoder` with `use_edges=False`, which drops the edge
predictor, the edge bias of the GNN attention, the edge update and the decoder edge slots, and leaves every other
line of code shared. With the same weights loaded and the edge bias zeroed, the graph model reproduces the set model
exactly, so a difference in the results can only come from the edges.
The DETR query layout is deliberately left untouched (`[global, edge context, nodes...]`), so the same `num_queries`
gives the same number of nodes in both arms; the edge context query is just never read.

On the default configuration, the edges are 5% of the parameters (136,837 of 2,711,874) but most of the compute: the
edge predictor runs `N^2 = 3844` times per image instead of `N = 62` times, and the latent holds 246,016 edge floats
against 3,968 node floats.


### Optimizers

The **AdamW** optimizer can be configured by setting the learning rate `lr`, the `weight_decay` and the `scheduler`.

The **SGD** optimizer can be configured by setting the learning rate `lr`, the `weight_decay`, the `momentum` and the `scheduler`.

The `weight_decay` is applied only to the weights of the convolutions, of the linear layers and of the embeddings:
every bias and every normalization scale is put in a group with no decay, since shrinking a LayerNorm scale or a bias
towards 0 costs accuracy on a deep transformer. With `weight_decay: 0.0` there is a single group, as before.

`grad_clip` clips the total gradient norm before every optimizer step (`0`, the default, disables it). A wide
transformer can spike in the first epochs; `1.0` is a safe value with pre-LN, and DETR uses `0.1` with post-LN.

The `scheduler` can be further customized, with the `scheduler_x` fields. `scheduler_exponential` allows `decay_rate`,
and `scheduler_cosine_with_warmup` allows `warmup_epochs`.
Currently, the scheduler updates the learning rate once per epoch, so `warmup_epochs` counts epochs and not steps.
With a dataset of tens of thousands of images that is fine: the first epoch already runs thousands of steps at a
fraction of the target learning rate.

```yaml
optimizer_adamw:
  lr: 1e-4
  weight_decay: 0.01
  grad_clip: 1.0
  scheduler: "constant" | "exponential" | "cosine" | "cosine_with_warmup"
  scheduler_cosine_with_warmup:
    warmup_epochs: 5
```

### Losses

The reconstruction loss functions can be configured within the field `reconstruction_x`, inside `loss`.

`l1`, `l2` and `bce` allows a `reduction` parameter, which can be set to `none`, `sum` or `mean`.

**Sparsity schedule.** `delay_epochs` and `ramp_epochs` (in `loss`) define a schedule with a progress from 0 (dense graph) to 1 (target sparsity):
it stays at 0 for `delay_epochs` epochs, then grows linearly over `ramp_epochs` epochs (`ramp_epochs: 0` jumps to 1).
The same schedule scales the gating thresholds of the model, the weights of the `probs` and `discr` regularizers, and the bounds of the `band` regularizer.
The progress is logged as `gating_warmup`, and the number of nodes and edges kept by the gating as `nodes_kept` and `edges_kept`.

**Band regularizer.** `nodes: "band"` and `edges: "band"` penalize an image if its number of kept nodes (or edges) is outside of `[min, max]`,
configured in `nodes_band` and `edges_band`. Inside the band there is no penalty, outside it grows linearly (in units of `max`, so nodes and edges have a similar scale).
The count is the one of the gating of the model (so `node_threshold` and `edge_threshold` must be greater than 0), and its gradient goes to all the confidences, also of the pruned elements.
The bounds follow the sparsity schedule: at progress 0 they are `[0, number of elements]` (no penalty), at progress 1 they are `[min, max]`.
The weights `beta` and `gamma` are not scaled by the schedule for this regularizer, because the bounds already are.
The edges are much more numerous than the nodes (N x N), so `gamma` usually needs to be much smaller than `beta`.

`hybrid` allows to set a transition of the value `alpha` (the relative weight between L1 and SSIM). The parameters `start_epoch`, `end_epoch`, `start_val`, `end_val` and `func` allows to set the transition.

```yaml
loss:
  reconstruction: "hybrid"
  nodes: "probs"
  edges: "probs"
  reconstruction_hybrid:
    start_epoch: 0
    end_epoch: 15
    start_val: 0.2
    end_val: 0.5
    func: "linear" | "cosine"
```

```yaml
loss:
  nodes: "band"
  edges: "band"
  nodes_band:
    min: 20
    max: 40
  edges_band:
    min: 20
    max: 40
  beta: 1.0
  gamma: 0.01
  delay_epochs: 10
  ramp_epochs: 10
```

### Inference Configuration

During inference, the same configuration file is used, but many parameters are ignored.

`model` selects which model implementation to use.

`checkpoint_file` selects which file to load the model from.

`loss` is still required, the inference script will compute the loss for the image it was executed on. For losses that changes over the epochs (tor example the `hybrid` loss with the transition), it is possible to set the epoch in which to compute the loss with `end_epoch`.