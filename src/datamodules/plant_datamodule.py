from typing import Optional, Tuple

import torch
import numpy as np 

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from sklearn.model_selection import KFold

from tqdm import tqdm 

from .datasets.plant_dataset import Plant
import json

from pathlib import Path
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
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
        # disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        #         '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        #         '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        #         '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        #         '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        #         '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}
        # risk = {'1':'초기','2':'중기','3':'말기'}

        # self.label_description = {}
        # for key, value in disease.items():
        #     self.label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
        #     for disease_code in value:
        #         for risk_code in risk:
        #             label = f'{key}_{disease_code}_{risk_code}'
        #             self.label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'
        
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
        return 32

    def corss_validation(self,foldn): 
        
        BASE_PATH = Path(self.hparams.data_dir).resolve()
        
        train_jpg = np.array(list(BASE_PATH.glob('*/*.jpg')))
        train_json = np.array(list(BASE_PATH.glob('*/*.json')))
        train_csv = np.array(list(BASE_PATH.glob('*/*.csv')))
        
        from joblib import Parallel, delayed
        # label_list consistance crops(bbox), risks, disesasse, labels 
        label_list = np.array(Parallel(n_jobs=32,prefer="threads")(delayed(self.json2cls)(i) for i in tqdm(train_json)))

        # label_unique = {key:idx for idx, key in enumerate(self.label_description)}
        label_unique = sorted(np.unique(label_list))
        label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

        self.label_decoder = {val:key for key, val in label_unique.items()}
        print({val:key for key, val in label_unique.items()})
        
        labels = [label_unique[k] for k in label_list[:,-1]]
        label_list[:,-1] = labels

        folds = []
        kf = KFold(n_splits=5, shuffle=True, random_state=2022)
        for train_idx, valid_idx in kf.split(train_jpg ):
            folds.append((train_idx, valid_idx))
        self.train_idx, self.valid_idx = folds[foldn]


        return train_jpg, train_csv,label_list


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        jpgpath, csvpath, labels = self.corss_validation(foldn = 0)
        
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = Plant(jpgpath[self.train_idx],labels[self.train_idx], mode='train', transform=self.transforms)
            self.data_val = Plant(jpgpath[self.valid_idx],labels[self.valid_idx], mode='valid', transform=self.transforms)

            test_path = Path(self.hparams.test_data_dir).resolve()
            test_jpg = np.array(list(test_path.glob('*/*.jpg')))
            self.data_test = Plant(test_jpg,None, mode='test', transform=self.transforms)
            
            # dataset = ConcatDataset(datasets=[trainset, testset])
            # self.data_train, self.data_val, self.data_test = random_split(
            #     dataset=dataset,
            #     lengths=self.hparams.train_val_test_split,
            #     generator=torch.Generator().manual_seed(2022),
            # )

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
