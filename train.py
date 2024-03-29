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


def main(args):
    if args.wandb.using:
        wandb.init(project=args.wandb.project, entity=args.wandb.entity)
        wandb.run.name = args.task_name
        wandb.util.generate_id()
    
    setSeeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    if args.use_kfold:
        trainer.run_kfold(args, train_data)
    else:
        train_data, valid_data = preprocess.split_data(train_data, ratio=args.split_ratio, seed=args.seed)
        trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--conf', default='/opt/ml/code/conf.yml', help='wrtie configuration file root.')
    # term_args = parser.parse_args()

    with open('/opt/ml/code/conf.yml') as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)
    args = AttrDict(cf)
    # args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
    
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