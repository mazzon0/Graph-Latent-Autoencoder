import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
import yaml
import sys

from models import get_model
from optimizers import get_lr_lambda, get_optimizer
from losses import get_loss

START_EPOCH = 0
END_EPOCH = 0
FROM_CHECKPOINT = False
CHECKPOINT_FILE = ''
BATCH_SIZE = 1

MODEL = None
OPTIMIZER = None
LOSS = None
MODEL_CONFIG = None
OPTIMIZER_CONFIG = None
LOSS_CONFIG = None

def load_config(filename: str):
    with open(filename, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
        if config:
            global START_EPOCH, END_EPOCH, FROM_CHECKPOINT, CHECKPOINT_FILE, BATCH_SIZE, MODEL, OPTIMIZER, LOSS, MODEL_CONFIG, OPTIMIZER_CONFIG, LOSS_CONFIG
            START_EPOCH = config.get('start_epoch', 0)
            END_EPOCH = config.get('end_epoch', 1)
            FROM_CHECKPOINT = config.get('from_checkpoint', False)
            CHECKPOINT_FILE = config.get('checkpoint_file', "")
            BATCH_SIZE = config.get('batch_size', 1)

            MODEL = config.get('model', "cnn")
            MODEL_CONFIG = config.get('model_' + MODEL, None)
            OPTIMIZER = config.get('optimizer', "adamw")
            OPTIMIZER_CONFIG = config.get('optimizer_' + OPTIMIZER, None)
            LOSS = config.get('loss', "cross_entropy")
            LOSS_CONFIG = config.get('loss_' + LOSS, None)

def collate_autoencoder(batch):
    # only returns the images
    images = [item[0] for item in batch]
    return images

def train():
    if torch.cuda.is_available():   print("Training on CUDA GPU")
    else:                           print("Training on CPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler('cuda')
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    # Data Preprocessing
    train_transform = v2.Compose([
        v2.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomHorizontalFlip(p=0.3),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    val_transform = v2.Compose([
        v2.Resize(64, antialias=True), 
        v2.CenterCrop(size=(64, 64)),  
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # Dataset
    train_set = CocoDetection(
        root='data/datasets/coco/train2017',
        annFile='data/datasets/coco/annotations/instances_train2017.json',
        transform=train_transform
    )

    val_set = CocoDetection(
        root='data/datasets/coco/val2017',
        annFile='data/datasets/coco/annotations/instances_val2017.json',
        transform=val_transform
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_autoencoder)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_autoencoder)

    # Initialize model, loss and optimizer
    model = get_model(MODEL, MODEL_CONFIG).to(device)
    loss_fn = get_loss(LOSS, LOSS_CONFIG).to(device)
    optimizer = get_optimizer(model, OPTIMIZER, OPTIMIZER_CONFIG)
    if FROM_CHECKPOINT:
        loaded_data = torch.load(CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(loaded_data['model_state_dict'], strict=True)
        optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
    
    lr_lambda_name = OPTIMIZER_CONFIG.get('lr_lambda', "constant")
    lr_lambda = get_lr_lambda(lr_lambda_name, OPTIMIZER_CONFIG.get("scheduler_" + lr_lambda_name, dict()), END_EPOCH)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=START_EPOCH-1)
    if FROM_CHECKPOINT:
        scheduler.load_state_dict(loaded_data['scheduler_state_dict'])
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(START_EPOCH, END_EPOCH):
        # Train
        model.train()
        train_losses = {'loss': 0.0}
        
        for images in train_loader:
            images = torch.stack(images).to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, images)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses['loss'] += loss

        # Validation
        model.eval()
        val_losses = {'loss': 0}

        with torch.no_grad():
            for images in val_loader:
                images = torch.stack(images).to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, images)
                val_losses['loss'] += loss
        
        # Save best and last model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_losses['loss']
        }
        torch.save(checkpoint, "last.pth")
        if val_losses['loss'] > best_loss:
            best_loss = val_losses['loss']
            torch.save(checkpoint, "best.pth")

        scheduler.step()

        avg_loss_train = train_losses['loss'] / len(train_loader)
        avg_loss_val = val_losses['loss'] / len(val_loader)

        print(f"Epoch {epoch}/{END_EPOCH}")
        print(f"Training losses: {avg_loss_train}")
        print(f"Validation losses: {avg_loss_val}")
        print(f"lr: {scheduler.get_last_lr()}")
        print("-" * 40)
        
if __name__ == '__main__':
    load_config(sys.argv[1])
    train()