import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class OCTDataset(Dataset):
    def __init__(self, data_dir, mode, classes, image_transforms):
        self.data_dir = data_dir
        self.image_transforms = image_transforms
        self.classes = classes
        self.images_path = []
        for c in classes:
            self.images_path.extend(glob.glob(os.path.join(data_dir, mode, c, '*.jpeg')))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image = Image.open(self.images_path[idx])
        image = self.image_transforms(image)
        label = 0
        for i, c in enumerate(self.classes):
            if c in os.path.basename(self.images_path[idx]):
                label = i
                break
        return image, label

def main():
    data_dir = 'D:/pytorch/Classification/OCT/data'
    mode = 'train'
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

if __name__ == '__main__':
    main()
