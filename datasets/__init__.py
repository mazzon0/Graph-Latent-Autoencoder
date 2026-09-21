import torch
from .coco import CocoDataset, load_resized

IMAGE_DIRS = {
    "coco": ('data/datasets/coco/train2017', 'data/datasets/coco/val2017'),
    "clevr": ('data/datasets/clevr/CLEVR_v1.0/images/train', 'data/datasets/clevr/CLEVR_v1.0/images/val'),
}

def get_dataset(name: str, image_size: int, train_transform, val_transform, cache: bool = True):
    train_dir, val_dir = IMAGE_DIRS[name]
    train_set = CocoDataset(train_dir, image_size, train_transform, cache)
    val_set = CocoDataset(val_dir, image_size, val_transform, cache)
    return train_set, val_set

def load_image(path: str, image_size: int) -> torch.Tensor:
    """Returns a float tensor (3, S, S) in [0, 1]."""
    return load_resized(path, image_size).float() / 255.0
