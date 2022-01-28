#!/bin/bash
cd ../
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python run.py trainer.gpus=7 +trainer.num_nodes=1 +trainer.strategy=ddp +trainer.precision=16 \
                                                    datamodule.data_dir=/nfs2/personal/cmpark/dacon/dataset/train \
                                                    datamodule.crop=False \
                                                    datamodule.batch_size=30 \
                                                    datamodule.label_type=positive \
                                                    model.model.num_classes=19 \
                                                    model.model.name=swin_large_patch4_window7_224_in22k
                                                    
                                                    