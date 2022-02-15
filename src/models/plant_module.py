
from typing import Any, List

import torch
import timm

from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, F1
from torchmetrics.classification.accuracy import Accuracy

import pandas as pd

import ttach as tta
# from src.models.components.simple_dense_net import SimpleDenseNet

# from ..utils.general import label_decoder, PlantCheckpointer
from ..utils.general import SAM, LabelSmoothingCrossEntropy,rand_bbox
import torch.nn.functional as F
from joblib import Parallel, delayed
import os
from pathlib import Path
import numpy as np 


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
        SAM: bool = False,
        label_smoothing: float = 0.05,
        cutmix : float = 0.05
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        model_parser = self.hparams.model
        
        self.first_test = True
        
        self.model = timm.create_model(model_parser.name, pretrained = model_parser.pretrained, num_classes = model_parser.num_classes)

        # if hasattr(model_parser, 'init_weight'):
        #     # if model_parser.init_weight in locals():
        #     # weight = torch.load(model_parser.init_weight)
        # #     self.model.load_state_dict(weight, strict=False)
        #     checkpointer = PlantCheckpointer(model=self.model)
        #     checkpointer.load(model_parser.init_weight)

        # if model_parser.init_weight in locals():
        #     checkpointer = PlantCheckpointer(model=self.model)
        #     checkpointer.load(model_parser.init_weight)

        self.SAM = self.hparams.SAM
        if self.SAM == True:
            self.automatic_optimization = False

        # loss function
        if self.hparams.label_smoothing == 0.0:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = LabelSmoothingCrossEntropy(epsilon=self.hparams.label_smoothing)

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

    def cutmix_step(self,batch:Any):        
        x = batch['img']
        y = batch['label']

        # generate mixed sample
        lam = np.random.beta(0.3, 0.3)
        rand_index = torch.randperm(x.size()[0]).cuda()
        target_a = y
        target_b = y[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

        logits = self.forward(x)
        loss = self.criterion(logits, target_a) * lam + self.criterion(logits, target_b) * (1. - lam)
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, y


    def step(self, batch: Any):
        x = batch['img']
        y = batch['label']

        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        if self.hparams.cutmix > 0.5:
            loss, preds, targets = self.cutmix_step(batch)
        else:
            loss, preds, targets = self.step(batch)

        if self.SAM == True:
            
            optimizer = self.optimizers()
            self.manual_backward(loss)
            optimizer.first_step(zero_grad=True)

            loss_2, _, _ = self.step(batch)
            self.manual_backward(loss_2)
            optimizer.second_step(zero_grad=True)

            self.trainer.train_loop.running_loss.append(loss)

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
        if self.first_test:
            self.model = tta.ClassificationTTAWrapper(self.model, tta.aliases.d4_transform())
            self.first_test = False
            
        preds = self.forward(batch['img'])
        prob = F.softmax(preds).max(1).values.cpu().numpy()
        preds = torch.argmax(preds, dim=1).cpu().detach().numpy()
        
        # lst = Parallel(n_jobs=16,prefer="threads")(delayed(self.write_csv)(i,j) for i,j in zip(batch[1],preds))
        for i,j,k in zip(batch['label'],preds, prob):
            self.write_csv(i,j,k)
        
        # self.submission.to_csv(f'sample{batch_idx}.csv')
        # return {"preds": preds,"idxs":batch[1]}
    def write_csv(self,idx,preds,prob):
        
        label_name = self.trainer.datamodule.label_decoder[preds]
        self.submission = self.submission.append({'image':int(idx),'label':str(label_name),'probability':str(prob)} , ignore_index=True)
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

    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_native_amp=False,
    #     using_lbfgs=False,
    # ):
    #     # skip the first 500 steps
    #     if self.trainer.global_step < 500:
    #         lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
    #         for pg in optimizer.param_groups:
    #             pg["lr"] = lr_scale * self.hparams.learning_rate

    #     # update params
    #     optimizer.step(closure=optimizer_closure)
        
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        if self.SAM == True:
            # base_optimizer = torch.optim.SGD
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(self.parameters(), base_optimizer, lr=self.hparams.lr)
            # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=5e-2)

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,35], gamma=0.1, last_epoch=-1, verbose=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=50, T_mult=2, eta_min=0,last_epoch=-1, verbose=False)
        
        # if self.trainer.global_epoch < 500:
        #     lr_scale = min(1.0, float(self.trainer.global_epoch + 1) / 500.0)
        #     for pg in optimizer.param_groups:
        #         pg["lr"] = lr_scale * self.hparams.lr
                
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=92 * 300 * 1.2) # epoch 25 step 92

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency":1,
            "monitor": "metric_to_track",
        }
        
        # return optimizer
        return {"optimizer":optimizer, "lr_scheduler":lr_scheduler_config}