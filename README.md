# Graph Latent Autoencoder
This project aims to train self-supervised Scene Graph Generation models and graph-conditioned image generation models.
This is achieved with an image-to-graph-to-image autoencoder.

## Table of Contents
* [Architecture](#architecture)
* [Quick Start](#quick-start)
    * [Setup](#setup)
    * [Training](#training)
    * [Inference](#inference)
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
git clone git@github.com:mazzon0/Graph-Latent-Autoencoder.git && cd Graph-Latent-Autoencoder
```

Create the virtual environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate
```

Download the COCO dataset.
```bash
./download.sh
```

### Training

Train the model (storing last and best model each epoch). The model, the loss, the optimizer, the lr scheduler and other parameters can be customized on the configuration file.
Before training, you need to activate the virtual environment with `source .venv/bin/activate`.
```bash
python3 train.py configs/some_configuration.yaml
```
Then, you can deactivate the virtual environment with `deactivate`.

### Inference

Test the model on a single image with the inference script. It will tell you the loss and the will save the original and reconstructed image side to side in `result.png`. You can use the same configuration file used for the training, specifying the model to use in the `from_checkpoint` field.
Before executing the model, you need to activate the virtual environment with `source .venv/bin/activate`.
```bash
python3 inference.py configs/some_configuration.yaml some_image.jpg
```
Then, you can deactivate the virtual environment with `deactivate`.

## Configuration

### General
All configuration files needs some general values.

`model` allows to select an autoencoder model, `optimizer` allows to select an optimizer and `dataset` allows to select a dataset.
`start_epoch` and `end_epoch` allows to choose the range of epochs for the training process. Epochs are 0 based. `start_epoch` and `end_epoch` are included in the range.

It is possible to initialize the model and the optimizer from a checkpoint, by setting `from_checkpoint` to `true` and specifying the path of the model in `checkpoint_file`. If `from_checkpoint` is `false`, then `checkpoint_file` is ignored.

It is possible to set the `batch_size` and `num_workers` values.

About the loss, it is possible to select which `reconstruction_loss` to use (`hybrid` is a linear combination of `l1` and `ssim`). It is possible to select the regularizers for the latent graph: the sum of the probabilities `probs` or the count of elements `discr`. The CNN Autoencoder will return 0 for these values. It is possible to select the weights of these losses `alpha`, `beta` and `gamma` respectively for reconstruction loss, nodes regularizer and edge regularizer.

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

The output images have values in the range [0, 1], so a sigmoid activation function is added at the end.
During training, for compatibility with the BCE loss, it is possible to return the actual logits,
by setting `train_with_sigmoid` to `false` (the sigmoid is applied only at inference).

```yaml
model_cnn:
  image_shape: [3, 64, 64]
  channels: [3, 64, 128, 256]
  mlp_sizes: [16384, 2048, 1024]
  train_with_sigmoid: true
```

### Optimizers

The **AdamW** optimizer can be configured by setting the learning rate `lr`, the `weight_decay` and the `scheduler`.

The **SGD** optimizer can be configured by setting the learning rate `lr`, the `weight_decay`, the `momentum` and the `scheduler`.

The `scheduler` can be further customized, with the `scheduler_x` fields. `scheduler_exponential` allows `decay_rate`,
and `scheduler_cosine_with_warmup` allows `warmup_epochs`.
Currently, the scheduler updates the learning rate once per epoch.

```yaml
optimizer_adamw:
  lr: 1e-4
  weight_decay: 0.0
  scheduler: "constant" | "exponential" | "cosine" | "cosine_with_warmup"
  scheduler_cosine_with_warmup:
    warmup_epochs: 5
```

### Losses

The reconstruction loss functions can be configured within the field `reconstruction_x`, inside `loss`.

`l1`, `l2` and `bce` allows a `reduction` parameter, which can be set to `none`, `sum` or `mean`.

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

### Inference Configuration

During inference, the same configuration file is used, but many parameters are ignored.

`model` selects which model implementation to use.

`checkpoint_file` selects which file to load the model from.

`loss` is still required, the inference script will compute the loss for the image it was executed on. For losses that changes over the epochs (tor example the `hybrid` loss with the transition), it is possible to set the epoch in which to compute the loss with `end_epoch`.