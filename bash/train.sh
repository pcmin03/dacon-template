#!bin/bash
cd ../
python run.py trainer.gpus=1 +trainer.precision=16 logger=wandb