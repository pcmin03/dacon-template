#!/bin/bash
cd ../
CUDA_VISIBLE_DEVICES=4,5,6,7 python run.py datamodule.batch_size=150 trainer.gpus=4 +trainer.num_nodes=1 +trainer.strategy=ddp +trainer.precision=16