import torch
from torchvision.transforms import v2
from torchvision.utils import save_image
import yaml
import sys
from PIL import Image

from models import get_model
from losses import get_loss

FROM_CHECKPOINT = True
CHECKPOINT_FILE = ''

MODEL = None
LOSS = None
MODEL_CONFIG = None

def load_config(filename: str):
    with open(filename, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
        if config:
            global CHECKPOINT_FILE, MODEL, LOSS, MODEL_CONFIG, LOSS_CONFIG
            CHECKPOINT_FILE = config.get('checkpoint_file', "")
            MODEL = config.get('model', "cnn")
            MODEL_CONFIG = config.get('model_' + MODEL, None)
            LOSS = config.get('loss', dict())

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
    if FROM_CHECKPOINT:
        loaded_data = torch.load(CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(loaded_data['model_state_dict'], strict=True)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(image)
        loss = loss_fn(output['image'], output['nodes'], output['edges'], image)

        output = model(image)
        reconstructed_image = output['image']
        loss = loss_fn(reconstructed_image, output['nodes'], output['edges'], image)

    # Result
    OUTPUT_PATH = 'result.png'
    comparison = torch.cat([image, reconstructed_image], dim=3)
    save_image(comparison, OUTPUT_PATH)
    
    print(f"Result saved in '{OUTPUT_PATH}'")
    print(f"Loss: {loss}")
        
if __name__ == '__main__':
    # Usage: python3 inference.py configs/config.yaml data/some_image.jpg
    if len(sys.argv) < 3:
        print("Usage: python3 inference.py <config_path> <image_path>")
    else:
        load_config(sys.argv[1])
        inference(sys.argv[2])