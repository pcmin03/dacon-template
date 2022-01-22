#!/bin/bash
cd ../
python run.py -m hparams_search=plant_optuna experiment=plant_example trainer.gpus=4 +trainer.num_nodes=1 +trainer.strategy=ddp +trainer.precision=16