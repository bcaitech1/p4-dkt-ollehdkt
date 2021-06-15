import os
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
import pandas as pd
import numpy as np

import torch
import pandas as pd
import numpy as np

# 미리 로딩
# args = parse_args(mode='train')
# device = "cuda" if torch.cuda.is_available() else "cpu"
# args.device = device
# args.n_questions = len(np.load(os.path.join(args.asset_dir,'assessmentItemID_classes.npy')))
# args.n_test = len(np.load(os.path.join(args.asset_dir,'testId_classes.npy')))
# args.n_tag = len(np.load(os.path.join(args.asset_dir,'KnowledgeTag_classes.npy')))
# model = trainer.load_model(args)
# model.to(device)
# args.model = model



def gen_data(data):
    df = pd.read_csv("questions.csv")
    new_columns = df.columns.tolist()+['answerCode']
    new_df = pd.DataFrame([],columns=new_columns+['userID'])
    
    for index, row in df.iterrows():
        user_actions = pd.DataFrame(data, columns=new_columns)    
        user_actions['userID'] = index
        new_df=new_df.append(user_actions)
        row['userID'] = index
        new_df=new_df.append(row)
    
    new_df['answerCode'].fillna(-1, inplace=True)
    new_df['answerCode']=new_df['answerCode'].astype(int)
    new_df['KnowledgeTag']=new_df['KnowledgeTag'].astype(str)
    new_df = new_df.reset_index(drop=True)
    return new_df

    
def inference(data,args):
    # 아래의 세 파일들은 준비되어 있어야함
    args.n_questions = 9455 
    args.n_test = 1538
    args.n_tag = 913
    # LSTM은 기본 64로 맞춰져 있다
    args.hidden_dim = 64
    args.max_seq_len = 20
    args.n_layers = 2
    args.n_heads = 2
    args.drop_out = 0.2

    args.cate_cols = ["testId",'assessmentItemID',"KnowledgeTag"]


    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    model = trainer.load_model(args)
    model = model.to(device)
    args.model = model
    print(model)
    
    print("Before:",data)
    data = gen_data(data)
    print("After:",data)
    print(args)
    
    #TODO
    #이곳에서 위에서 생성한 데이터를 기반으로 inference한 값들을 평균을 내서 입력해주시면 되겠습니다.
    #probability의 평균
    preprocess = Preprocess(args)
    preprocess.load_test_data(data)
    
    test_data = preprocess.get_test_data()
    
    result = trainer.inference_for_serving(args, test_data)

    return result    

