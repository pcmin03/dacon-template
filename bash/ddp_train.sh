#!/bin/bash
cd ../
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py datamodule.batch_size=8 trainer.gpus=8 +trainer.num_nodes=1 +trainer.strategy=ddp +trainer.precision=16 \
                                                    datamodule.data_dir=/nfs2/personal/cmpark/dacon/dataset/train \
                                                    datamodule.crop=False \
                                                    datamodule.batch_size=200 \
                                                    datamodule.label_type=positive \
                                                    model.model.num_classes=20 \
                                                    model.model.name=efficientnet_b2
                                                    
                                                    