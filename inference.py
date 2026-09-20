import torch
from torchvision.transforms import v2
from torchvision.utils import save_image
import yaml
import os
import sys
from PIL import Image

from models import get_model, apply_sigmoid_if_requested
from losses import get_loss
from utils.experiment import resolve_checkpoint

FROM_CHECKPOINT = True
CHECKPOINT_FILE = ''
EXPERIMENT_NAME = None
APPLY_SIGMOID = True
END_EPOCH = 0

MODEL = None
LOSS = None
MODEL_CONFIG = None

def load_config(filename: str):
    with open(filename, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
        if config:
            global APPLY_SIGMOID, CHECKPOINT_FILE, EXPERIMENT_NAME, MODEL, LOSS, MODEL_CONFIG, LOSS_CONFIG, END_EPOCH
            END_EPOCH = config.get('end_epoch', END_EPOCH)
            CHECKPOINT_FILE = config.get('checkpoint_file', "")
            EXPERIMENT_NAME = config.get('experiment_name', None)
            MODEL = config.get('model', "cnn")
            MODEL_CONFIG = config.get('model_' + MODEL, None)
            LOSS = config.get('loss', dict())
            APPLY_SIGMOID = MODEL_CONFIG.get('apply_sigmoid', True)

def inference(image_path: str):
    if torch.cuda.is_available():   print("Inference on CUDA GPU")
    else:                           print("Inference on CPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Preprocessing
    val_transform = v2.Compose([
        v2.Resize(64, antialias=True), 
        v2.CenterCrop(size=(64, 64)),  
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # Load image
    img_raw = Image.open(image_path).convert('RGB')
    image = val_transform(img_raw).unsqueeze(0).to(device)

    # Initialize model, loss and optimizer
    model = get_model(MODEL, MODEL_CONFIG).to(device)
    loss_fn = get_loss(LOSS).to(device)
    output_dir = "."
    if FROM_CHECKPOINT:
        checkpoint_path = resolve_checkpoint(CHECKPOINT_FILE, EXPERIMENT_NAME)
        output_dir = os.path.join(os.path.dirname(checkpoint_path), "inference")
        loaded_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(loaded_data['model_state_dict'], strict=True)
    
    # Inference
    inspector_payload = None
    model.eval()
    with torch.no_grad():
        output = model(image)                                               # image logits
        loss_dict = loss_fn(apply_sigmoid_if_requested(output, APPLY_SIGMOID), image, END_EPOCH)
        
        if hasattr(model, 'export_for_inspector'):
            inspector_payload = model.export_for_inspector(image, batch_idx=0)

    # Result Outputs
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    OUTPUT_IMAGE_PATH = os.path.join(output_dir, image_name + '.png')
    OUTPUT_DATA_PATH = os.path.join(output_dir, image_name + '.pt')
    
    # Save reconstructed image comparison
    comparison = torch.cat([image, torch.sigmoid(output['image'])], dim=3)    # the saved image always needs values in [0, 1]
    save_image(comparison, OUTPUT_IMAGE_PATH)
    print(f"Result image saved in '{OUTPUT_IMAGE_PATH}'")
    
    # Save data for Inspector
    if inspector_payload is not None:
        torch.save(inspector_payload, OUTPUT_DATA_PATH)
        print(f"Scene graph data saved in '{OUTPUT_DATA_PATH}'")
    
    # Metrics breakdown
    print(f"Total Loss: {loss_dict['loss'].item():.4f}")
    print("Loss breakdown:")
    for key, val in loss_dict.items():
        if key != 'loss':
            print(f"  - {key}: {val.item():.4f}")
        
if __name__ == '__main__':
    # Usage: python3 inference.py configs/config.yaml data/some_image.jpg
    if len(sys.argv) < 3:
        print("Usage: python3 inference.py <config_path> <image_path>")
    else:
        load_config(sys.argv[1])
        inference(sys.argv[2])