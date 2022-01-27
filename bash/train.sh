#!bin/bash
cd ../
CUDA_VISIBLE_DEVICES=4 python run.py trainer.gpus=1 datamodule.batch_size=1 +trainer.precision=16 logger=wandb \
                                                    datamodule.data_dir=/pathos2/data2/dongheekim/Plant/dacon-template/data/train \
                                                    datamodule.crop=True