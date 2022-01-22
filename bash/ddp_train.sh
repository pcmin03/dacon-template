#!/bin/bash
cd ../
python run.py datamodule.batch_size=200 trainer.gpus=8 +trainer.num_nodes=1 +trainer.strategy=ddp +trainer.precision=16