from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np

class Plant(Dataset):
    def __init__(self, images, labels, mode='train',transform=None):
        
        self.img_paths = images
        self.labels = labels
        self.mode = mode

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        img = cv2.imread(str(self.img_paths[idx]))[...,::-1]
        img = cv2.resize(img, (384, 512))

        if self.mode=='train':  
          img = transforms.ToTensor()(img)

        elif self.mode=='valid':  
          img = transforms.ToTensor()(img)

        elif self.mode=='test':
            return transforms.ToTensor()(img),self.img_paths[idx].name('.jpg','')
        
        label = int(self.labels[idx][-1])
        
        return img, label