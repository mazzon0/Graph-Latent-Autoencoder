import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import yaml
import sys
import os
import time

from models import get_model, apply_sigmoid_if_requested
from optimizers import get_lr_lambda, get_optimizer
from losses import get_loss
from datasets import get_dataset
from utils.schedule import warmup_progress
from utils.experiment import create_experiment_dir, resolve_checkpoint, log_metrics

START_EPOCH = 0
END_EPOCH = 0
FROM_CHECKPOINT = False
CHECKPOINT_FILE = ''
EXPERIMENT_NAME = None
APPLY_SIGMOID = True
CONFIG_PATH = ''
BATCH_SIZE = 1
NUM_WORKERS = 1
CACHE_DATASET = True
IMAGE_SIZE = 64

MODEL = None
OPTIMIZER = None
LOSS = None
MODEL_CONFIG = None
OPTIMIZER_CONFIG = None
LOSS_CONFIG = None
DATASET = None

torch.manual_seed(0)

def load_config(filename: str):
    with open(filename, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
        if config:
            global APPLY_SIGMOID, START_EPOCH, END_EPOCH, FROM_CHECKPOINT, CHECKPOINT_FILE, EXPERIMENT_NAME, BATCH_SIZE, NUM_WORKERS, CACHE_DATASET, IMAGE_SIZE, MODEL, OPTIMIZER, LOSS, MODEL_CONFIG, OPTIMIZER_CONFIG, LOSS_CONFIG, DATASET
            START_EPOCH = config.get('start_epoch', 0)
            END_EPOCH = config.get('end_epoch', 1)
            FROM_CHECKPOINT = config.get('from_checkpoint', False)
            CHECKPOINT_FILE = config.get('checkpoint_file', "")
            EXPERIMENT_NAME = config.get('experiment_name', None)
            BATCH_SIZE = config.get('batch_size', 1)
            NUM_WORKERS = config.get('num_workers', 1)
            CACHE_DATASET = config.get('cache_dataset', True)

            MODEL = config.get('model', "cnn")
            MODEL_CONFIG = config.get('model_' + MODEL, None)
            OPTIMIZER = config.get('optimizer', "adamw")
            OPTIMIZER_CONFIG = config.get('optimizer_' + OPTIMIZER, None)
            LOSS = config.get('loss', dict())
            APPLY_SIGMOID = MODEL_CONFIG.get('apply_sigmoid', True)
            image_shape = MODEL_CONFIG.get('image_shape', [3, 64, 64])
            assert image_shape[1] == image_shape[2], f"the dataset pipeline produces square images, but image_shape is {image_shape}"
            IMAGE_SIZE = image_shape[1]
            DATASET = config.get('dataset', "coco")

def collate_autoencoder(batch):
    # only returns the images
    images = torch.stack([item[0] for item in batch])
    return images

def train():
    if torch.cuda.is_available():   print("Training on CUDA GPU")
    else:                           print("Training on CPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Experiment directory (resumed runs reuse the same directory)
    resume = FROM_CHECKPOINT and EXPERIMENT_NAME is not None and os.path.isdir(os.path.join("experiments", EXPERIMENT_NAME))
    experiment_dir = create_experiment_dir(CONFIG_PATH, MODEL, EXPERIMENT_NAME, resume)
    print(f"Experiment directory: {experiment_dir}")

    # Data Preprocessing
    # The dataset already stores the images resized and center cropped to IMAGE_SIZE (uint8)
    train_transform = v2.Compose([
        v2.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.5, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomHorizontalFlip(p=0.3),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToDtype(torch.float32, scale=True),
    ])

    val_transform = v2.ToDtype(torch.float32, scale=True)

    # Dataset
    train_set, val_set = get_dataset(DATASET, IMAGE_SIZE, train_transform, val_transform, CACHE_DATASET)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_autoencoder, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_autoencoder, prefetch_factor=4)

    # Initialize model, loss and optimizer
    model = get_model(MODEL, MODEL_CONFIG).to(device)
    loss_fn = get_loss(LOSS).to(device)
    optimizer = get_optimizer(model, OPTIMIZER, OPTIMIZER_CONFIG)
    best_loss = float('inf')
    if FROM_CHECKPOINT:
        loaded_data = torch.load(resolve_checkpoint(CHECKPOINT_FILE, EXPERIMENT_NAME), map_location=device)
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
    grad_norms = torch.zeros(num_batches, device=device)
    for epoch in range(START_EPOCH, END_EPOCH + 1):
        start = time.time()

        # Sparsity schedule (the same used by the loss)
        model.set_sparsity_progress(warmup_progress(epoch, LOSS.get('delay_epochs', 0), LOSS.get('ramp_epochs', 0)))

        # Train
        model.train()
        train_losses = dict()
        
        for i, batch in enumerate(train_loader):
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)

            outputs = apply_sigmoid_if_requested(model(images), APPLY_SIGMOID)
            
            loss = loss_fn(outputs, images, epoch)

            optimizer.zero_grad()
            loss['loss'].backward()
            grad_norms[i] = model.get_first_layer().weight.grad.norm()
            optimizer.step()

            for key, l in loss.items():
                if key in train_losses:
                    train_losses[key] += l.detach()
                else:
                    train_losses[key] = l.detach()

        avg_norm = torch.mean(grad_norms).item()
        std_norm = torch.std(grad_norms).item()
        median_norm = torch.median(grad_norms).item()
        
        train_losses = {k: v.item() / num_batches for k, v in train_losses.items()}

        # Validation
        model.eval()
        val_losses = dict()

        with torch.no_grad():
            for batch in val_loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(device, non_blocking=True)
                
                outputs = apply_sigmoid_if_requested(model(images), APPLY_SIGMOID)
                loss = loss_fn(outputs, images, epoch)
                
                for key, l in loss.items():
                    if key in val_losses:
                        val_losses[key] += l.detach()
                    else:
                        val_losses[key] = l.detach()
                        
        val_losses = {k: v.item() / len(val_loader) for k, v in val_losses.items()}

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_losses['loss']
        }
        torch.save(checkpoint, os.path.join(experiment_dir, "last.pth"))
        if val_losses['loss'] < best_loss:
            print("New best model")
            best_loss = val_losses['loss']
            torch.save(checkpoint, os.path.join(experiment_dir, "best.pth"))

        scheduler.step()

        log_metrics(experiment_dir, {
            'epoch': epoch,
            'seconds': time.time() - start,
            'lr': scheduler.get_last_lr()[0],
            'train': train_losses,
            'val': val_losses,
        })

        print(f"Epoch {epoch}/{END_EPOCH}  -  {time.time() - start:.2f} seconds")
        print(f"First Layer Grad Norms: median = {median_norm:.4f}, mean = {avg_norm:.4f}, std = {std_norm:.4f}")
        print(f"Training losses: {train_losses}")
        print(f"Validation losses: {val_losses}")
        print(f"lr: {scheduler.get_last_lr()}")
        print("-" * 80)
        
if __name__ == '__main__':
    CONFIG_PATH = sys.argv[1]
    load_config(CONFIG_PATH)
    train()