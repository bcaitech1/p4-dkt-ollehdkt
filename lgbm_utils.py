import json
import pandas as pd
import os
import random
from attrdict import AttrDict
# !pip install ligthgbm 
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
from pre_FE import *

def make_lgbm_feature(df,is_train=True):
    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    
    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    correct_k.columns = ["tag_mean", 'tag_sum']
    #학생의 학년을 정하고 푼 문제지의 학년합을 구해본다
    df['test_level']=df['assessmentItemID'].apply(lambda x:int(x[2]))
    correct_l = df.groupby(['userID'])['test_level'].agg(['mean', 'sum'])
    correct_l.columns = ["level_mean", 'level_sum']
    df = pd.merge(df, correct_l, on=['userID'], how="left")

    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")

    df.isnull().sum()
    df = df.fillna(0)
    return df

def lgbm_split_data(data,ratio):
    random.seed(42)
    #load & apply pre-extracted feature
    # data['distance']=np.load('/opt/ml/np_train_tag_distance_arr.npy')
    data['total_tag_ansrate']=np.load('/opt/ml/np_train_total_tag_ansrate_arr.npy')
    data['user_tag_ansrate']=np.load('/opt/ml/np_train_user_tag_ansrate_arr.npy')    

    users = list(zip(data['userID'].value_counts().index, data['userID'].value_counts()))
    random.shuffle(users)
    
    max_train_data_len = ratio*len(data)
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)


    train = data[data['userID'].isin(user_ids)]
    test = data[data['userID'].isin(user_ids) == False]

    #test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test

def lgbm_make_test_data(data):
    data = data.drop(['answerCode'], axis=1)
    return data[data['userID'] != data['userID'].shift(-1)]

def lgbm_train(args,train_data,valid_data):
    # 사용할 Feature 설정
    delete_feats=['userID','assessmentItemID','testId','answerCode','Timestamp']

    FEATS = list(set(train_data.columns)-set(delete_feats))
    print(f'{len(FEATS)}개의 피처를 사용하여 학습합니다')
    print(FEATS)
    # FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
    #          'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']

    # X, y 값 분리
    y_train = train_data['answerCode']
    train_data = train_data.drop(['answerCode'], axis=1)

    y_test = valid_data['answerCode']
    valid_data = valid_data.drop(['answerCode'], axis=1)

    lgb_train = lgb.Dataset(train_data[FEATS], y_train)
    lgb_test = lgb.Dataset(valid_data[FEATS], y_test)
    model = lgb.train(
#                     {'objective': 'binary'}, 
                args.lgbm.model_params,
                lgb_train,
                valid_sets=[lgb_train, lgb_test],
                verbose_eval=args.lgbm.verbose_eval, #ori 100
                num_boost_round= args.lgbm.num_boost_round,
                early_stopping_rounds=args.lgbm.early_stopping_rounds,
            )

    preds = model.predict(valid_data[FEATS])
    acc = accuracy_score(y_test, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_test, preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')
    
    ax = lgb.plot_importance(model)
    #Feature Importance 저장
    new_output_path=f'{args.output_dir}{args.task_name}'
    write_path = os.path.join(new_output_path, "feature_importance.png")
    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path) 
    ax.figure.savefig(write_path)

    return model,auc,acc
    
def lgbm_inference(args,model, test_data):
    
    delete_feats=['userID','assessmentItemID','testId','answerCode','Timestamp']
    FEATS = list(set(test_data.columns)-set(delete_feats))
    print(f'{len(FEATS)}개의 피처를 사용하여 추론합니다')
    print(FEATS)
    answer = model.predict(test_data[FEATS])
    
    # 테스트 예측 결과 저장
    new_output_path=f'{args.output_dir}{args.task_name}'
    write_path = os.path.join(new_output_path, "output.csv")
    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(answer):
            w.write('{},{}\n'.format(id,p))

    print(f"lgbm의 예측파일이 {new_output_path}/{args.task_name}.csv 로 저장됐습니다.")

    save_path=f"{args.output_dir}{args.task_name}/feature{len(FEATS)}_config.json"
    json.dump(
        FEATS,
        open(save_path, "w"),
        indent=2,
        ensure_ascii=False,
    )