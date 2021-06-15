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
from server import main as s_main
from test import main as test_main

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='./conf.yml', help='wrtie configuration file root.')
    term_args = parser.parse_args()

    with open(term_args.conf) as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)
    args = AttrDict(cf)
    # args.concat_reverse = False # 범주형, 연속형 순으로 합침(False), 그 반대 True
    # args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    
    #run_server
    if args.work == 'server':
        args.cate_cols = ['assessmentItemID','testId','KnowledgeTag',"test_level"]
        args.cont_cols = ['solve_time', 'mean_elapsed', 'test_time', 'grade_time', # 시간
            'answer_acc','tag_acc', 'test_acc', 'assess_acc', # 정답률
            'level_tag_acc', 'level_test_acc', 'level_assess_acc', # 대분류&정답률
            'tag_cumAnswer',]
        args.n_cate_cols = len(args.cate_cols)
        args.n_cont_cols = len(args.cont_cols)
        d= {} # key는 컬럼명, values는 임베딩 시킬 수

        for i in args.cate_cols:
            if i=='grade':
                d[i] = 9
            elif i == 'rank_point':
                d[i]=10000
            else:
                d[i] = len(np.load(os.path.join(args.asset_dir,f'{i}_classes.npy')))
        print(f'임베딩 사이즈 확인')
        print(d)
        args.cate_dict = d
        s_main(args)
    elif args.work == 'test':
        args.cate_cols = ['assessmentItemID','testId','KnowledgeTag',"test_level"]
        args.cont_cols = ['solve_time', 'mean_elapsed', 'test_time', 'grade_time', # 시간
            'answer_acc','tag_acc', 'test_acc', 'assess_acc', # 정답률
            'level_tag_acc', 'level_test_acc', 'level_assess_acc', # 대분류&정답률
            'tag_cumAnswer',]
        args.n_cate_cols = len(args.cate_cols)
        args.n_cont_cols = len(args.cont_cols)
        d= {} # key는 컬럼명, values는 임베딩 시킬 수

        for i in args.cate_cols:
            if i=='grade':
                d[i] = 9
            elif i == 'rank_point':
                d[i]=10000
            else:
                d[i] = len(np.load(os.path.join(args.asset_dir,f'{i}_classes.npy')))
        print(f'임베딩 사이즈 확인')
        print(d)
        args.cate_dict = d
        test_main(args)
    else:
        #train
        if args.run_train:
            t_main(args)
        #inference
        if args.run_inference:
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