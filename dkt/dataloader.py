import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from makefeature import *
from .features import Features as fe

n_test_level_diff=10000
n_unique = 0

from lgbm_utils import *


class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None
        

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data


    def split_data(self, data, ratio=0.7, shuffle=True, seed=42):
        """
        split data into two parts with a given ratio.
        """
        #lgbm일 경우
        if self.args.model=='lgbm':
            return lgbm_split_data(data,ratio,seed)

        #lgbm이 아닐 경우
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy') # 해당 클래스는 numpy로 저장
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train = True):
        #범주형 column 골라내기
        cate_cols=[]
        cont_cols=[]
        for column in df.columns:
            if df[column].dtype==object:
                cate_cols.append(column)
            else :
                cont_cols.append(column)
                

        #feat이름 배열 conf에 저장 
        self.args.cate_feats=cate_cols
        self.args.cont_feats=cont_cols
        
        
        print(f"범주형의 개수는 {len(cate_cols)}개 이고, 연속형의 개수는 {len(cont_cols)}개 입니다")
        print(f'범주형 {cate_cols}')
        print(f'연속형 {cont_cols}')
        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols[1:]:
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                le.classes_ = np.load(label_path)
                
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
            #모든 컬럼이 범주형이라고 가정
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test
            
        #cate feat들의 이름 / 고유값 개수를 dict로 conf에 저장
        self.args.cate_feat_dict=dict(zip(cate_cols,[len(df[col].unique()) for col in cate_cols]))
        return df

    def __feature_engineering(self, df):
        #TODO
        if self.args.model=='lgbm':
            return make_lgbm_feature(self.args, df)
        else:

            #lgbm 외의 다른 모델들의 fe가 필요하다
            # df = fe.feature_engineering_03(df) # 종호님 피쳐는 먼저 나와야한다.
            df=make_feature(self.args,df)
            # df = df.merge(fe.feature_engineering_06(pd.DataFrame(df)), left_index=True,right_index=True, how='left')
            print(f'FE후 컬럼 확인 : {df.columns}')
            print(df.columns)

            print('dataframe 확인')
            print(df)

            # drop_cols = ['_',"index","point","answer_min_count","answer_max_count","user_count",'sec_time'] # drop할 칼럼
            # for col in drop_cols:
            #     if col in df.columns:
            #         df.drop([col],axis=1, inplace=True)
            # print(f"drop 후 : {df.columns}")

            delete_feats=['Timestamp','sec_time']
            df=df.drop(columns=delete_feats)
            features = df.columns
            print(f"drop 후 : {len(features)}개, {features}")
            return df
        

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        print(f'csv_file_path : {csv_file_path}')
        df = pd.read_csv(csv_file_path)
        print("load data 전 유저 수",len(df['userID'].unique()))
        if self.args.model=='lgbm':
    
            df.sort_values(by=['userID','Timestamp'], inplace=True)
            return df

        if is_train and self.args.user_split_augmentation:
            #종호님의 유저 split augmentation
            df['Timestamp']=pd.to_datetime(df['Timestamp'].values)
            df['month'] = df['Timestamp'].dt.month
            # df['userID'] = (df['userID'].map(str)+'0'+df['month'].map(str)).astype('int32')
            df['userID'] = df['userID'].map(str)+'-'+df['month'].map(str)
            df.drop(columns=['month'],inplace=True)
            print("user_augmentation 후 유저 수",len(df['userID'].unique()))
        
        #둘은 int로 돼있어서 cate_col로 분류되도록 미리 형변환
        df['userID']=df['userID'].astype(str)
        df['KnowledgeTag']=df['KnowledgeTag'].astype(str)
        

        col_cnt = len(df.columns)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # df['userID']=df['userID'].astype(int)
        df['KnowledgeTag']=df['KnowledgeTag'].astype(int)
        

        #column은 cate feats 다음에 cont_feats가 오며 cate feats의 처음은 userid, cont_feats의 처음 피처는 answerCode임
        columns=self.args['cate_feats']+self.args['cont_feats']
        print(columns)
        #기존 피처 유저제외시킴
        ret = columns[1:]
        #연속형 첫번째 순서인 answerCode를 빼서
        ret.pop(len(self.args.cate_feats)-1)
        #맨뒤로 붙여줌
        ret.append('answerCode')
        print("ret",ret)
        print("answercode의 순서 뒤로 변경",ret)
        group = df[columns].groupby('userID').apply(
                lambda r: tuple([r[i].values for i in ret])
            )
        print(group)
        print(f"유저수 {len(group)} 피처수 {len(group.iloc[0])} 푼 문제 수 {len(group.iloc[0][0])}")
        len(f'group.values->{len(group.values)}')
        print("load data 후",len(df['userID'].unique()))
        return group.values, pd.DataFrame(df['userID'].unique(), columns=['userID'])


    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name,is_train=False)

class MyDKTDataset(torch.utils.data.Dataset):
    def __init__(self,data, args):
        self.data = data
        self.args = args

    def __getitem__(self,index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        # print(f'row 값 : {len(row)}')

        # test, question, tag, correct, solve_time...etc
        columns = [row[i] for i in range(len(row))]
  
        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(columns):
                columns[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        columns.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(columns):
            columns[i] = torch.tensor(col)

        return columns

    def __len__(self):
        return len(self.data)

class TestDKTDataset(torch.utils.data.Dataset):
    def __init__(self,data, args):
        self.data = data
        self.args = args

    def __getitem__(self,index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        print(f'row 값 : {len(row)}')

        # test, question, tag, correct, solve_time
        cate_cols = [row[i] for i in range(len(row))]
        

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col.astype(int))

        return cate_cols

    def __len__(self):
        return len(self.data)
        


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct = row[0], row[1], row[2], row[3]
        

        cate_cols = [test, question, tag, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                #가장 최근 것부터 적용, 예전 기록들은 지운다
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            #0으로 패딩
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    #마스크의 길이로 max_seq_len
    max_seq_len = len(batch[0][-1])

        
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            #앞부분에 마스킹을 넣어주어 sequential하게 interaction들을 학습하게 한다
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)


    for i, _ in enumerate(col_list):
        #stack을 통해 피처 텐서를 이어붙인다(차원축으로) <-> torch.cat
        #각 배치에서 shape(len(feature),len(max_seq_len)) -> shape(len(feature),1,len(max_seq_len)) 
        col_list[i] =torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = True
    train_loader, valid_loader = None, None
    
    if train is not None:
        # trainset = DKTDataset(train, args)
        # trainset = DevDKTDataset(train,args)
        # trainset = TestDKTDataset(train,args)
        trainset = MyDKTDataset(train,args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = MyDKTDataset(valid,args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader


def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = args.stride

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1
            
            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(col[window_i*stride:window_i*stride + window_size])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))


    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas


def data_augmentation(data, args):
    if args.window == True:
        data = slidding_window(data, args)

    return data

