from torch.utils.data import Dataset
from PIL import Image
import os
from config import label_map
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class CustomTxtDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.samples = []
        self.transform = transform
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                path, label_str = line.strip().rsplit(',', 1)
                label = label_map[label_str]
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
