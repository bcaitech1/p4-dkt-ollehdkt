import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
from attrdict import AttrDict
import yaml
import json
from train import main as t_main
from inference import main as i_main

if __name__ == "__main__":
    cf = yaml.load(open('/opt/ml/code/conf.yml'))
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
    json.dump(
        args,
        open(save_path, "w"),
        indent=2,
        ensure_ascii=False,
    )