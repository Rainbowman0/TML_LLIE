import os
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MyDS(Dataset):
    def __init__(self, root, size=(400, 640)):
        self.size = size
        self.data_names = glob(os.path.join(root, '*.jpg')) + glob(os.path.join(root, '*.png')) + glob(os.path.join(root, '*.jpeg'))

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        img_path = self.data_names[idx]
        sample = Image.open(img_path)
        sample = sample.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        sample = transform(sample)

        return sample, img_path