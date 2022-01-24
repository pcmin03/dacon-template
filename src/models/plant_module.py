from typing import Any, List

import torch
import timm

from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, F1
from torchmetrics.classification.accuracy import Accuracy

import pandas as pd
# from src.models.components.simple_dense_net import SimpleDenseNet

from ..utils.general import label_decoder, PlantCheckpointer
import torch.nn.functional as F 
from joblib import Parallel, delayed

class PlantCls(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: Any, 
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()
        # self.label_decoder = {0: '0', 1: '00', 2: '1', 3: '1_00_0', 4: '2', 5: '2_00_0', 6: '2_a5_2', 7: '3', 8: '3_00_0', 9: '3_a9_1', 10: '3_a9_2', 11: '3_a9_3', 12: '3_b3_1', 13: '3_b6_1', 14: '3_b7_1', 15: '3_b8_1', 16: '4', 17: '4_00_0', 18: '5', 19: '5_00_0', 20: '5_a7_2', 21: '5_b6_1', 22: '5_b7_1', 23: '5_b8_1', 24: '6', 25: '6_00_0', 26: '6_a11_1', 27: '6_a11_2', 28: '6_a12_1', 29: '6_a12_2', 30: '6_b4_1', 31: '6_b4_3', 32: '6_b5_1', 33: 'a11', 34: 'a12', 35: 'a5', 36: 'a7', 37: 'a9', 38: 'b3', 39: 'b4', 40: 'b5', 41: 'b6', 42: 'b7', 43: 'b8'}
        # self.label_decoder = self.trainer.datamodule
        
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        model_parser = self.hparams.model
        
        self.model = timm.create_model(model_parser.name,pretrained = model_parser.pretrained, num_classes = model_parser.num_classes)
        # import pdb;pdb.set_trace()
        if model_parser.init_weight is not None:
            weight = torch.load(model_parser.init_weight)
            self.model.load_state_dict(weight, strict=False)
            # checkpointer = PlantCheckpointer(model=self.model)
            # checkpointer.load(model_parser.init_weight)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.train_f1 = F1(num_classes = model_parser.num_classes, average='macro')
        self.val_f1 = F1(num_classes = model_parser.num_classes, average='macro')
        self.test_f1 = F1(num_classes = model_parser.num_classes, average='macro')

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()
        # self.submission = pd.read_csv('/nfs2/personal/cmpark/dacon/dataset/sample_submission.csv')
        self.submission = pd.DataFrame(columns=['image','label'])

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        f1 = self.train_f1(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):

        loss, preds, targets = self.step(batch)
        # log val metrics
        acc = self.val_acc(preds, targets)
        f1 = self.val_f1(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        f1 = self.val_f1.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.val_f1_best.update(f1)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/f1_best", self.val_f1_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        preds = self.forward(batch[0])
        preds = torch.argmax(preds, dim=1).cpu().detach().numpy()
        
        # lst = Parallel(n_jobs=16,prefer="threads")(delayed(self.write_csv)(i,j) for i,j in zip(batch[1],preds))
        for i,j in zip(batch[1],preds):
            self.write_csv(i,j)
        # self.submission.to_csv(f'sample{batch_idx}.csv')
        return {"preds": preds,"idxs":batch[1]}
    def write_csv(self,idx,preds):
        label_name = self.trainer.datamodule.label_decoder[preds] 
        # df=df.append({'image' : 'Apple' , 'label' : 23} , ignore_index=True)
        self.submission = self.submission.append({'image':int(idx),'label':str(label_name)} , ignore_index=True)
        # self.submission.loc[self.submission.image == int(idx),'label'] = str(label_name)
        # print(self.submission.loc[self.submission.image == int(idx),'label'])
    def test_epoch_end(self, outputs: List[Any]):
        
        self.submission = self.submission.sort_values(by='image')
        self.submission.to_csv(f'sampleas123df.csv', index=False)
        
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

        self.train_f1.reset()
        self.test_f1.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,80], gamma=0.1, last_epoch=-1, verbose=False)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency":1,
        }
        
        return optimizer
        # return {"optimizer":optimizer, "lr_scheduler":lr_scheduler_config}
