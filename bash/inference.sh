#!bin/bash
cd ../
python run.py training=False datamodule.batch_size=400 trainer.gpus=1 