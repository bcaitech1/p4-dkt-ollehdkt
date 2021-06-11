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