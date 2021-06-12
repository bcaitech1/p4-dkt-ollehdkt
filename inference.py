import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
import yaml
import argparse
from attrdict import AttrDict

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
#     args.infer = True
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()

    if args.use_kfold:
      trainer.inference_kfold(args, test_data)
    else :
      trainer.inference(args, test_data)
      

if __name__ == "__main__":

    with open('/opt/ml/code/conf.yml') as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)
    args = AttrDict(cf)
    # args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)