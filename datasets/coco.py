import os
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def load_resized(path: str, image_size: int) -> torch.Tensor:
    """Loads an image, resizes the short side to `image_size` and center crops it. Returns a uint8 tensor (3, S, S)."""
    img = Image.open(path)
    img.draft('RGB', (image_size * 2, image_size * 2))
    img = img.convert('RGB')

    w, h = img.size
    scale = image_size / min(w, h)
    new_w, new_h = max(image_size, round(w * scale)), max(image_size, round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC, reducing_gap=None)

    left, top = (new_w - image_size) // 2, (new_h - image_size) // 2
    img = img.crop((left, top, left + image_size, top + image_size))
    return torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(image_size, image_size, 3).permute(2, 0, 1)

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, image_size, transform=None, cache=True):
        self.transform = transform
        self.image_size = image_size
        if cache:
            self.data = self._load_or_build(img_dir, image_size)
            self.data.share_memory_()   # each data loader workers would otherwise get one copy
            self.paths = None
        else:
            self.data = None
            self.paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]

    @staticmethod
    def _load_or_build(img_dir: str, image_size: int) -> torch.Tensor:
        img_names = sorted(os.listdir(img_dir))
        cache_path = f"{os.path.normpath(img_dir)}.pt"

        if os.path.exists(cache_path):
            data = torch.load(cache_path)
            if data.shape[0] == len(img_names) and data.shape[-1] == image_size:
                return data
            print(f"'{cache_path}' is out of date (different number of images or image size), rebuilding it")

        print(f"Resizing {len(img_names)} images of '{img_dir}' to {image_size}x{image_size} (only done once)...")
        paths = [os.path.join(img_dir, f) for f in img_names]
        data = torch.empty(len(paths), 3, image_size, image_size, dtype=torch.uint8)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            for i, img in enumerate(pool.map(lambda p: load_resized(p, image_size), paths)):
                data[i] = img
                if (i + 1) % 10000 == 0:
                    print(f"  {i + 1}/{len(paths)}")

        torch.save(data, cache_path)
        return data

    def __len__(self):
        return len(self.paths) if self.data is None else self.data.shape[0]

    def __getitem__(self, idx):
        if self.data is None:
            img = load_resized(self.paths[idx], self.image_size)
        else:
            img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, 0
