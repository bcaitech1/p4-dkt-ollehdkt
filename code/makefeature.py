import warnings
warnings.filterwarnings('ignore')

from matplotlib import  rc
import matplotlib.pyplot as plt
import json
import pandas as pd
import os,gc
import random
from attrdict import AttrDict
# !pip install ligthgbm 
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GroupKFold
from sklearn.impute import SimpleImputer
import numpy as np
from collections import defaultdict

from lgbm_utils import *

def make_feature(args,df):
    testId_mean_sum, assessmentItemID_mean_sum, KnowledgeTag_mean_sum,testId_time_agg,assessment_time_agg,KnowledgeTag_time_agg,assess_time_corNwrong_agg=get_sharing_feature(args)

    #문제별 맞은사람과 틀린사람의 평균풀이시간
    df['wrongP_time']=df.assessmentItemID.map(assess_time_corNwrong_agg['first'])
    df['correctP_time']=df.assessmentItemID.map(assess_time_corNwrong_agg['last'])
    
    item_size = df[['assessmentItemID', 'testId']].drop_duplicates().groupby('testId').size()
    testId2maxlen = item_size.to_dict() # 중복해서 풀이할 놈들을 제거하기 위해

    df["test_mean"] = df.testId.map(testId_mean_sum['mean'])
    df['test_sum'] = df.testId.map(testId_mean_sum['sum'])
    df["ItemID_mean"] = df.assessmentItemID.map(assessmentItemID_mean_sum['mean'])
    df['ItemID_sum'] = df.assessmentItemID.map(assessmentItemID_mean_sum['sum'])
    df["tag_mean"] = df.KnowledgeTag.map(KnowledgeTag_mean_sum['mean'])
    df['tag_sum'] = df.KnowledgeTag.map(KnowledgeTag_mean_sum['sum'])  

    df['test_t_mean']= df.testId.map(testId_time_agg['mean'])
    df['test_t_std']= df.testId.map(testId_time_agg['std'])
    df['test_t_skew']= df.testId.map(testId_time_agg['skew'])
    df['assess_t_mean']= df.assessmentItemID.map(assessment_time_agg['mean'])
    df['assess_t_std']= df.assessmentItemID.map(assessment_time_agg['std'])
    df['assess_t_skew']= df.assessmentItemID.map(assessment_time_agg['skew'])
    df['tag_t_mean']= df.KnowledgeTag.map(KnowledgeTag_time_agg['mean'])
    df['tag_t_std']= df.KnowledgeTag.map(KnowledgeTag_time_agg['std'])
    df['tag_t_skew']= df.KnowledgeTag.map(KnowledgeTag_time_agg['skew'])
    ###서일님 피처

    df['test_tag_cumsum']=df.groupby(['userID','testId']).agg({'solve_time':'cumsum'})
    
    # 유저가 푼 시험지에 대해, 유저의 전체 정답/풀이횟수/정답률 계산 (3번 풀었으면 3배)
    df_group = df.groupby(['userID','testId'])['answerCode']
    df['user_total_correct_cnt'] = df_group.transform(lambda x: x.cumsum().shift(1))
    df['user_total_ans_cnt'] = df_group.cumcount()
    df['user_total_acc'] = df['user_total_correct_cnt'] / df['user_total_ans_cnt']

    #학생의 학년을 정하고 푼 문제지의 학년합을 구해본다
    df['test_level']=df['assessmentItemID'].apply(lambda x:str(x[2]))
    #문제번호
    df['problem_number']=df['assessmentItemID'].apply(lambda x:str(x[-3:]))

    group_list=['userID']
    uid_agg_dict={
        'solve_time' :['mean','std','skew'],
    }
    
    agg_dict_list=[uid_agg_dict]
    
    
    for group, now_agg in zip(group_list,agg_dict_list):
        grouped_df=df.groupby(group).agg(now_agg)
        new_cols = []
        for col in now_agg.keys():
            for stat in now_agg[col]:
                if type(stat) is str:
                    new_cols.append(f'{group}-{col}-{stat}')
                else:
                    new_cols.append(f'{group}-{col}-mode')

        grouped_df.columns = new_cols

        grouped_df.reset_index(inplace = True)
        df = df.merge(grouped_df, on=group, how='left')
    #delete null
    df.isnull().sum()
    df = df.fillna(0)
    return df