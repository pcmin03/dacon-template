#!bin/bash
cd ../
CUDA_VISIBLE_DEVICES=4 python run.py trainer.gpus=1 +trainer.precision=16 logger=wandb