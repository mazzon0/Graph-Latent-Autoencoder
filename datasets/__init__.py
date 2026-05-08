from .coco import CocoDataset

def get_dataset(name: str, train_transform, val_transform):
    match name:
        case "coco":
            train_set = CocoDataset(
                img_dir='data/datasets/coco/train2017',
                transform=train_transform
            )

            val_set = CocoDataset(
                img_dir='data/datasets/coco/val2017',
                transform=val_transform
            )
    
    return train_set, val_set