#!/bin/bash
# checkpath = "/nfs2/personal/cmpark/dacon/dacon-template/logs/runs/2022-01-19/05-55-05/checkpoints/epoch_008.ckpt"
cd ../
python run.py training=False trainer.gpus=4 +trainer.num_nodes=1 +trainer.strategy=ddp +trainer.precision=16 +resume_from_checkpoint="/nfs2/personal/cmpark/dacon/dacon-template/logs/runs/2022-01-19/05-55-05/checkpoints/epoch_008.ckpt"