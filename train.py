import os
from args import parse_args
from dkt.dataloader import Preprocess,data_augmentation
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
import yaml
import json
import argparse
from attrdict import AttrDict

import mlflow

def main(args):
    if args.wandb.using:
        wandb.init(project=args.wandb.project, entity=args.wandb.entity)
        wandb.run.name = args.task_name
        wandb.util.generate_id()

    if args.mlflow.using:
        print(f"Use MLflow OK... for exp_name : {args.mlflow.experiment}")
        try:
            #프로젝트 별로 이름을 다르게 가져가면서 실험들을 기록
            mlflow.create_experiment(name=args.mlflow.experiment)
            
        except:
            print('Exist experiment')
        # mlflow.create_experiment(name=args.mlflow.experiment)
        mlflow.set_experiment(args.mlflow.experiment)
        #mlflow에 기록할 준비
        mlflow.start_run()
        mlflow.set_tag('version', '0.1')

        # 모델 하이퍼 파라미터 설정
        params = {
            'learning_rate' : args.lr,
            'n_epochs' : args.n_epochs,
            'batch_size' : args.batch_size
        }

        mlflow.set_experiment(args.mlflow.experiment)
        mlflow.log_params(params)
    
    setSeeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(args.device)

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data, train_data_userID_df = preprocess.get_train_data()

    if args.use_kfold:
        trainer.run_kfold(args, train_data, train_data_userID_df)
    else:
        
        print(len(train_data))
        
        train_data, valid_data = preprocess.split_data(train_data, ratio=args.split_ratio, seed=args.seed)

        train_data = data_augmentation(train_data,args)
        trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--conf', default='/opt/ml/code/conf.yml', help='wrtie configuration file root.')
    # term_args = parser.parse_args()

    with open('/opt/ml/git/dh_branch/p4-dkt-ollehdkt/conf.yml') as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)
    args = AttrDict(cf)
    # args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
    
    # args.pop('wandb')
    
    # save_path=f"{args.output_dir}{args.task_name}/exp_config.json"
    # if args.model=='lgbm':
    #     args=args.lgbm
        
    # else :
    #     args.pop('lgbm')
    # json.dump(
    #     args,
    #     open(save_path, "w"),
    #     indent=2,
    #     ensure_ascii=False,
    # )
