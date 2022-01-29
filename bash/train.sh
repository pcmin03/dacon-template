#!bin/bash
cd ../
CUDA_VISIBLE_DEVICES=4 python run.py trainer.gpus=1 datamodule.batch_size=1 +trainer.precision=16 logger=wandb \
                                                    datamodule.data_dir=/pathos2/nfs2/personal/cmpark/dacon/dataset/train \
                                                    datamodule.label_type=total model.lr=1e-4 \
                                                    model.model.num_classes=25
                                                    # datamodule.crop=True