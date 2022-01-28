from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
class Plant(Dataset):
    def __init__(self, images, labels, mode='train',transform=None):
        
        self.img_paths = images
        self.labels = labels
        self.mode = mode

        self.transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5)
        # A.RandomBrightnessContrast(p=0.2),
    ])

#         self.transform = transform


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        img = cv2.imread(str(self.img_paths[idx]))[...,::-1]

        img = cv2.resize(img, (384, 384))

        if self.mode=='train':  
          result = self.transform(image=img)
          img = transforms.ToTensor()(result['image'])

        if self.mode=='train':  
          img = self.transform(img)

        elif self.mode=='valid':  
          img = self.transform(img)

        elif self.mode=='test':
          img = transforms.ToTensor()(img)
          
          return img,self.img_paths[idx].name.replace('.jpg','')
        
        label = int(self.labels[idx][-1])
        
        return img, label