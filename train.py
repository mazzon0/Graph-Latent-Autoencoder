import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import yaml
import sys
import time

from models import get_model
from optimizers import get_lr_lambda, get_optimizer
from losses import get_loss
from datasets import get_dataset

START_EPOCH = 0
END_EPOCH = 0
FROM_CHECKPOINT = False
CHECKPOINT_FILE = ''
BATCH_SIZE = 1
NUM_WORKERS = 1

MODEL = None
OPTIMIZER = None
LOSS = None
MODEL_CONFIG = None
OPTIMIZER_CONFIG = None
LOSS_CONFIG = None
DATASET = None

def load_config(filename: str):
    with open(filename, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
        if config:
            global START_EPOCH, END_EPOCH, FROM_CHECKPOINT, CHECKPOINT_FILE, BATCH_SIZE, NUM_WORKERS, MODEL, OPTIMIZER, LOSS, MODEL_CONFIG, OPTIMIZER_CONFIG, LOSS_CONFIG, DATASET
            START_EPOCH = config.get('start_epoch', 0)
            END_EPOCH = config.get('end_epoch', 1)
            FROM_CHECKPOINT = config.get('from_checkpoint', False)
            CHECKPOINT_FILE = config.get('checkpoint_file', "")
            BATCH_SIZE = config.get('batch_size', 1)
            NUM_WORKERS = config.get('num_workers', 1)

            MODEL = config.get('model', "cnn")
            MODEL_CONFIG = config.get('model_' + MODEL, None)
            OPTIMIZER = config.get('optimizer', "adamw")
            OPTIMIZER_CONFIG = config.get('optimizer_' + OPTIMIZER, None)
            LOSS = config.get('loss', dict())
            DATASET = config.get('dataset', "coco")

def collate_autoencoder(batch):
    # only returns the images
    images = torch.stack([item[0] for item in batch])
    return images

def train():
    if torch.cuda.is_available():   print("Training on CUDA GPU")
    else:                           print("Training on CPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    train_set, val_set = get_dataset(DATASET, train_transform, val_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_autoencoder, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_autoencoder, prefetch_factor=4)

    # Initialize model, loss and optimizer
    model = get_model(MODEL, MODEL_CONFIG).to(device)
    loss_fn = get_loss(LOSS).to(device)
    optimizer = get_optimizer(model, OPTIMIZER, OPTIMIZER_CONFIG)
    best_loss = float('inf')
    if FROM_CHECKPOINT:
        loaded_data = torch.load(CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(loaded_data['model_state_dict'], strict=True)
        optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
        best_loss = float(loaded_data.get('loss', float('inf')))
    
    lr_lambda_name = OPTIMIZER_CONFIG.get('scheduler', "constant")
    lr_lambda = get_lr_lambda(lr_lambda_name, OPTIMIZER_CONFIG.get("scheduler_" + lr_lambda_name, dict()), END_EPOCH)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=START_EPOCH-1)
    if FROM_CHECKPOINT:
        scheduler.load_state_dict(loaded_data['scheduler_state_dict'])
    
    # Training loop
    num_batches = len(train_loader)
    grad_norms = torch.zeros(num_batches)
    for epoch in range(START_EPOCH, END_EPOCH + 1):
        start = time.time()

        # Train
        model.train()
        train_losses = dict()
        
        for i, images in enumerate(train_loader):
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            loss = loss_fn(outputs['image'], outputs['nodes'], outputs['edges'], images, epoch)

            optimizer.zero_grad()
            loss['loss'].backward()
            grad_norms[i] = model.get_first_layer().weight.grad.norm()
            optimizer.step()

            for key, loss in loss.items():
                if key in train_losses:
                    train_losses[key] += loss.item()
                else:
                    train_losses[key] = loss.item()

        avg_norm = torch.mean(grad_norms).item()
        std_norm = torch.std(grad_norms).item()
        median_norm = torch.median(grad_norms).item()

        # Validation
        model.eval()
        val_losses = dict()

        for images in val_loader:
            images = images.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = loss_fn(outputs['image'], outputs['nodes'], outputs['edges'], images, epoch)
            
            for key, loss in loss.items():
                if key in val_losses:
                    val_losses[key] += loss.item()
                else:
                    val_losses[key] = loss.item()
        
        # Save best and last model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_losses['loss']
        }
        torch.save(checkpoint, "last.pth")
        if val_losses['loss'] < best_loss:
            print("New best model")
            best_loss = val_losses['loss']
            torch.save(checkpoint, "best.pth")

        scheduler.step()

        train_losses = {k: v / len(train_loader) for k, v in train_losses.items()}
        val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}
        print(f"Epoch {epoch}/{END_EPOCH}  -  {time.time() - start:.2f} seconds")
        print(f"First Layer Grad Norms: median = {median_norm:.4f}, mean = {avg_norm:.4f}, std = {std_norm:.4f}")
        print(f"Training losses: {train_losses}")
        print(f"Validation losses: {val_losses}")
        print(f"lr: {scheduler.get_last_lr()}")
        print("-" * 80)
        
if __name__ == '__main__':
    load_config(sys.argv[1])
    train()