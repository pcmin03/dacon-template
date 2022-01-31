#!/bin/bash
cd ../
# python run.py -m hparams_search=plant_optuna experiment=plant_example trainer.gpus=4 +trainer.nu
python run.py -m hparams_search=plant_optuna experiment=plant_example \
                datamodule.batch_size=10 trainer.gpus=6 +trainer.num_nodes=1 +trainer.strategy=ddp +trainer.precision=32 \
                datamodule.data_dir=/nfs2/personal/cmpark/dacon/dataset/train \
                datamodule.label_type=total datamodule.fold=$fold \
                model.model.name=swin_large_patch4_window12_384_in22k \
                model.model.num_classes=25 model.lr=1e-2 trainer.max_epochs=100 +model.SAM=True +model.label_smoothing=0.05 \
                +model.cutmix=0.5
