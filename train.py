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

def main(args):
    wandb.init(project=args.wandb.project, entity=args.wandb.entity)
    wandb.run.name = args.task_name
    
    setSeeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    
    train_data, valid_data = preprocess.split_data(train_data)
    
    
    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    with open('/opt/ml/code/conf.yml') as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)
    args = AttrDict(cf)
    # args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
    
    args.pop('wandb')
    args.pop('lgbm')
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