import os
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

class GeorgeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data.iloc[idx, 0])
        image = Image.open(image_path).convert('RGB')
        label = self.data.iloc[idx]['label']

        if transforms:
            image = self.transform(image)

        return image, label