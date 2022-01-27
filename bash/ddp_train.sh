#!/bin/bash
cd ../
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py datamodule.batch_size=8 trainer.gpus=8 +trainer.num_nodes=1 +trainer.strategy=ddp +trainer.precision=16 \
                                                    datamodule.data_dir=/pathos2/data2/dongheekim/Plant/dacon-template/data/train \
                                                    datamodule.crop=True