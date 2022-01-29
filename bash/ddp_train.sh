#!/bin/bash
cd ../
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python run.py trainer.gpus=7 +trainer.num_nodes=1 +trainer.strategy=ddp +trainer.precision=16 \
#                                                     datamodule.data_dir=/nfs2/personal/cmpark/dacon/dataset/train \
#                                                     datamodule.crop=False \
#                                                     datamodule.batch_size=30 \
#                                                     datamodule.label_type=positive \
#                                                     model.model.num_classes=19 \
#                                                     model.model.name=swin_large_patch4_window7_224_in22k

for fold in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py datamodule.batch_size=8 trainer.gpus=8 +trainer.num_nodes=1 +trainer.strategy=ddp +trainer.precision=32 \
                                                        datamodule.data_dir=/pathos2/nfs2/personal/cmpark/dacon/dataset/train \
                                                        datamodule.label_type=total datamodule.fold=$fold \
                                                        model.model.num_classes=25 model.lr=1e-2 trainer.max_epochs=100 model.SAM=True model.label_smoothing=0.05
                                                        # datamodule.crop=True \
                                                        # /pathos2/data2/dongheekim/Plant/dacon-template/data/train
done
                                                    