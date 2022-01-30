from typing import Optional, Tuple

import torch
import numpy as np 

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from sklearn.model_selection import KFold,StratifiedKFold

from tqdm import tqdm 

from .datasets.plant_dataset import Plant
import json

from pathlib import Path
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

class PlantModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        test_data_dir : str = 'data/',
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        crop: bool = False,
        training:bool = True,
        label_type: str = 'total',
        fold: int = 0
    ):
        super().__init__()
        self.total_label = {'1_00_0': 0,'2_00_0': 1,'2_a5_2': 2,'3_00_0': 3,'3_a9_1': 4,
            '3_a9_2': 5,'3_a9_3': 6,'3_b3_1': 7,'3_b6_1': 8,'3_b7_1': 9,
            '3_b8_1': 10,'4_00_0': 11,'5_00_0': 12,'5_a7_2': 13,'5_b6_1': 14,
            '5_b7_1': 15,'5_b8_1': 16,'6_00_0': 17,'6_a11_1': 18,'6_a11_2': 19,
            '6_a12_1':20,'6_a12_2':21,'6_b4_1': 22,'6_b4_3': 23,'6_b5_1': 24
                }

        self.binary_label = {'1_00_0': 0,'2_00_0': 0,'2_a5_2': 1,'3_00_0': 0,'3_a9_1': 1,
            '3_a9_2': 1,'3_a9_3': 1,'3_b3_1': 1,'3_b6_1': 1,'3_b7_1': 1,
            '3_b8_1': 1,'4_00_0': 0,'5_00_0': 0,'5_a7_2': 1,'5_b6_1': 1,
            '5_b7_1': 1,'5_b8_1': 1,'6_00_0': 0,'6_a11_1': 1,'6_a11_2': 1,
            '6_a12_1': 1,'6_a12_2': 1,'6_b4_1': 1,'6_b4_3': 1,'6_b5_1': 1
                }
        self.positive_label = {'1_00_0': 0,'2_00_0': 0,'2_a5_2': 1,'3_00_0': 0,'3_a9_1': 2,
            '3_a9_2': 3,'3_a9_3': 4,'3_b3_1': 5,'3_b6_1': 6,'3_b7_1': 7,
            '3_b8_1': 8,'4_00_0': 0,'5_00_0': 0,'5_a7_2': 9,'5_b6_1': 10,
            '5_b7_1': 11,'5_b8_1': 12,'6_00_0': 0,'6_a11_1': 13,'6_a11_2': 14,
            '6_a12_1': 15,'6_a12_2': 16,'6_b4_1': 17,'6_b4_3': 18,'6_b5_1': 19
                }

        self.negative_label = {'1_00_0': 1,'2_00_0': 2,'2_a5_2': 0,'3_00_0': 3,'3_a9_1': 0,
            '3_a9_2': 0,'3_a9_3': 0,'3_b3_1': 0,'3_b6_1': 0,'3_b7_1': 0,
            '3_b8_1': 0,'4_00_0': 4,'5_00_0': 5,'5_a7_2': 0,'5_b6_1': 0,
            '5_b7_1': 0,'5_b8_1': 0,'6_00_0': 6,'6_a11_1': 0,'6_a11_2': 0,
            '6_a12_1': 0,'6_a12_2': 0,'6_b4_1': 0,'6_b4_3': 0,'6_b5_1': 0
                }

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        # data transformations
        self.train_transforms = A.Compose(
            [
                # transforms.Resize((224,224)),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomSizedCrop(min_max_height=(224,384),height=384,width=384),
                A.Cutout(always_apply=True),
                A.CLAHE(p=0.5),
                # A.ColorJitter(p=0.5),
                A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ToTensor(),
            ]
        )
    
        self.test_transform = A.Compose(
            [
                # transforms.Resize((224,224)),
                A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ToTensor(),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.label_type = label_type
        self.fold = fold
        self.training = training
        
    def json2cls(self,jsonfile): 
        label_type = []
        with open(str(jsonfile), 'r') as f:
            sample = json.load(f)
            crop = sample['annotations']['crop']
            disease = sample['annotations']['disease']
            risk = sample['annotations']['risk']
            label=f"{crop}_{disease}_{risk}"

        return [crop,disease,risk,label]


    @property
    def num_classes(self) -> int:
        return len(self.label_decoder)

    def cross_validation(self,foldn): 
        
        BASE_PATH = Path(self.hparams.data_dir).resolve()
        
        if self.hparams.crop == True:
            train_jpg = np.array(list(BASE_PATH.glob('img_crop/*.jpg')))
            train_json = np.array(list(BASE_PATH.glob('json/*.json')))
        else:
            train_jpg = np.array(list(BASE_PATH.glob('*/*.jpg')))
            train_json = np.array(list(BASE_PATH.glob('*/*.json')))
        train_csv = np.array(list(BASE_PATH.glob('*/*.csv')))
        
        from joblib import Parallel, delayed
        # label_list consistance crops(bbox), risks, disesasse, labels 
        label_list = np.array(Parallel(n_jobs=32,prefer="threads")(delayed(self.json2cls)(i) for i in tqdm(train_json)))
        
        if self.label_type == 'total': 
            labels = np.array([self.total_label[k] for k in label_list[:,-1]])
            label_convert = self.total_label
        elif self.label_type == 'binary': 
            labels = np.array([self.binary_label[k] for k in label_list[:,-1]])
            label_convert = self.binary_label
        else:
            if self.label_type == 'positive': 
                labels = np.array([self.positive_label[k] for k in label_list[:,-1]])
                label_convert = self.positive_label
            elif self.label_type == 'negative': 
                labels = np.array([self.negative_label[k] for k in label_list[:,-1]])
                label_convert = self.negative_label
            
            # filtering label zero image
            _,locat = np.unique(labels,return_inverse=True)
            
            train_jpg = train_jpg[locat!=0]
            label_list = label_list[locat!=0]
            labels = np.array(labels[locat!=0]) - 1
            
            label_convert = {val-1:key for key,val in label_convert.items() if key!= 0}

        label_list[:,-1] = labels

        self.label_decoder = {val:key for key, val in label_convert.items()}

        folds = []

        # kf = KFold(n_splits=5, shuffle=True, random_state=2022)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)

        for train_idx, valid_idx in kf.split(train_jpg,np.array(label_list[:,-1])):
            folds.append((train_idx, valid_idx))

        self.train_idx, self.valid_idx = folds[foldn]
        
        return train_jpg, train_csv, label_list


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        jpgpath, csvpath, labels = self.cross_validation(foldn = self.fold)
        
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = Plant(jpgpath[self.train_idx],labels[self.train_idx], mode='train', transform=self.train_transforms)
            self.data_val = Plant(jpgpath[self.valid_idx],labels[self.valid_idx], mode='valid', transform=self.test_transform)

            # if self.training  == True : 
            #     self.data_test = self.data_val
            # else : 
            test_path = Path(self.hparams.test_data_dir).resolve()
            test_jpg = np.array(list(test_path.glob('*/*.jpg')))
            self.data_test = Plant(test_jpg,None, mode='test', transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
