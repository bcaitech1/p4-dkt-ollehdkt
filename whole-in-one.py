import os
import yaml
import json
import argparse
from attrdict import AttrDict

from dkt.dataloader import Preprocess
from dkt import trainer
from dkt.utils import setSeeds

import torch
import wandb

from train import main as t_main
from inference import main as i_main

if __name__ == "__main__":

    with open('/opt/ml/code/conf.yaml') as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)
    args = AttrDict(cf)
    
    # args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)

    #train
    t_main(args)
    #inference
    i_main(args)


    #save config_file as light_version
    args.pop('wandb')
    
    save_path=f"{args.output_dir}{args.task_name}/exp_config.json"
    if args.model=='lgbm':
        args=args.lgbm
    else :
        args.pop('lgbm')

    json.dump(
        args,
        open(save_path, "w"),
        indent=2,
        ensure_ascii=False,
    )