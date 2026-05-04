# Graph Latent Autoencoder
This project aims to train self-supervised Scene Graph Generation models and graph-conditioned image generation models.
This is achieved with an image-to-graph-to-image autoencoder.

## Setup

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

## Training

Train the model (storing last and best model each epoch).
```bash
source .venv/bin/activate
python3 train.py configs/some_configuration.yaml
deactivate
```
