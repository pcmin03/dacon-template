#!/bin/bash
cd ../
python run.py logger=wandb trainer.gpus=4 +trainer.num_nodes=2 +trainer.strategy=ddp +trainer.precision=16