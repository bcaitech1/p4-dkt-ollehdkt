import os
import yaml
import argparse
from attrdict import AttrDict


from dkt.dataloader import Preprocess
from dkt import trainer
import torch


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()
    

    trainer.inference_kfold(args, test_data)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='/opt/ml/git/p4-dkt-ollehdkt/conf.yml', help='wrtie configuration file root.')
    term_args = parser.parse_args()

    with open(term_args.conf) as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)
    args = AttrDict(cf)
    # args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)