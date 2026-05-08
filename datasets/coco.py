import os
import torch
from PIL import Image

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_names = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]    # images are cached
        self.transform = transform
        self.cache = {}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx], 0
        
        img = Image.open(self.img_names[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        self.cache[idx] = img
        return img, 0