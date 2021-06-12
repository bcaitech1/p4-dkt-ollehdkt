import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

from .features import Features as fe

n_test_level_diff=10000
n_unique = 0

from lgbm_utils import *

import gc

n_cate_cols = 0
n_cont_cols = 0
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
        
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag']

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
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
            

        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        df['Timestamp'] = df['Timestamp'].apply(convert_time)
        
        return df

    def __preprocessing_v2(self, df, is_train = True):
        
        cate_cols = self.args.cate_cols # numpy로 mapping 시킬 범주형 컬럼

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
            print(f'컬럼 : {col}')
            if col in ['grade']: # grade는 모두 9로 매핑하는 버그가 있어 제외시킨다
                continue
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                if col == 'solve_time':
                    a = [i for i in range(3601)] + ['unknown']
                else:
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
            
        print(df)
        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        df['Timestamp'] = df['Timestamp'].apply(convert_time)
        
        return df

    def __feature_engineering(self, df):
        #TODO
        if self.args.model=='lgbm':
            return make_lgbm_feature(self.args, df)
        else:
            #lgbm 외의 다른 모델들의 fe가 필요하다
            print(f'feature engineering for Transformers...')
            # df = fe.feature_engineering_03(df) # 종호님 피쳐는 먼저 나와야한다.
            # print(f'{df.columns}')
            # df = fe.feature_engineering_13(df)
            # df = fe.feature_engineering_07(df)
            # df = fe.feature_engineering_08(df)
            # df = fe.feature_engineering_09(df)
            # df = fe.feature_engineering_10(df)
            # df = fe.feature_engineering_11(df)
            # df = fe.feature_engineering_12(df)

            # df = df.merge(fe.feature_engineering_06(pd.DataFrame(df)), left_index=True,right_index=True, how='left')
            # df = fe.feature_engineering_06(df)
            print(f'fe 시 컬럼 확인 : {df.columns}')
            print(df.columns)

            if self.args.file_name == 'train_inter_time (1).csv' or self.args.file_name == 'test_inter_time (1).csv'\
            or self.args.file_name == 'train_time_finalfix.csv' or self.args.file_name == 'test_time_finalfix.csv':
                # df['hour'] = df.sec_time.apply(lambda x : f'h-{x//3600}')
                # df['min'] = df.sec_time.apply(lambda x : f'm-{(x%3600)//60}')
                # df['sec'] = df.sec_time.apply(lambda x : f's-{x%60}')
                
                df['solve_time'] = df['solve_time'].apply(lambda x : int(round(x)))
                
            df = fe.feature_engineering_14(df)
            print('dataframe 확인')
            print(df)

            drop_cols = ['_',"index","point","answer_min_count","answer_max_count","user_count","sec_time"] # drop할 칼럼
            for col in drop_cols:
                if col in df.columns:
                    df.drop([col],axis=1, inplace=True)
            print(f"drop 후 : {df.columns}")

            return df

    def load_data_from_file_v4(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        print(f'csv_file_path : {csv_file_path}')
        df = pd.read_csv(csv_file_path)

        col_cnt = len(df.columns)
        df = self.__feature_engineering(df)

        # ※ 주의 : gsaintplus 이용 시 밤주형만 입력할 것
        self.args.cate_cols = ['assessmentItemID','testId','KnowledgeTag'] # 실험할 범주형
        self.args.cont_cols = [] # 실험할 연속형 (user_acc)'solve_time', 'user_acc','user_correct_answer', 'user_total_answer',
        df = self.__preprocessing_v2(df, is_train)

        # 유효 컬럼만 거르기 (Optional)
        self.args.cate_cols = [col for col in list(self.args.cate_cols) if col in df.columns]
        self.args.cont_cols = [col for col in list(self.args.cont_cols) if col in df.columns]

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용 
        
        d= {} # key는 컬럼명, values는 임베딩 시킬 수

        for i in self.args.cate_cols:
            if i=='grade':
                d[i] = 9
            else:
                d[i] = len(np.load(os.path.join(self.args.asset_dir,f'{i}_classes.npy')))
        print(f'임베딩 사이즈 확인')
        print(d)
        self.args.cate_dict = d
        # user_correct_answer, user_total_answer,user_acc
        print('컬럼 확인')
        print(df.columns)

        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag',] # 기본 train_data에 있는 컬럼이므로 고정

        # user_count 기준으로 other feature를 구성

        # columns.extend(['test_level_diff','tag_sum','tag_mean','ans_rate'])
        columns.extend(list(self.args.cate_cols))
        columns.extend(list(self.args.cont_cols))
        self.args.n_other_features = [ int(df[i].nunique()) for i in df.columns[col_cnt:]] # 컬럼 순서 꼭 맞출 것!, 추가 컬럼(feature)의 고윳값 수

        # 데이터 프레임 재정비
        ret = []
        
        print(df)
        ret.extend(list(self.args.cate_cols))
        ret.append('answerCode')
        ret.extend(self.args.cont_cols)

        # answerCode 컬럼 없으면 강제 종료
        if 'answerCode' not in ret:
            import sys
            print("plz add column, answerCode...")
            sys.exit()
            return

        # new_df = df['userID']
        # # new_df = pd.concat([df[i] for i in self.args.cate_cols],axis=1)
        # for i in ret:
        #     new_df[i] = df[i]
        # df = new_df
        # gc.collect()

        print('보낼 최종컬럼 확인')
        print(ret)
        # print(df[columns])
        group = df.groupby('userID').apply(
                lambda r: tuple([r[i].values for i in ret])
            )
        
        print(f'group.values->{len(group.values)}')
        print(group)

        del df
        gc.collect()
        return group.values # 보낼 컬럼 = (범주형)  + (정답 여부) + (연속형)


    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)#, nrows=100000)
        
        if self.args.model=='lgbm':
            #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
            df.sort_values(by=['userID','Timestamp'], inplace=True)
            return df

        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)
       
        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용     

        # default 태그
        self.args.n_test_level_diff = 2563 # 변별력 고유값 개수 : 2563
        self.args.n_ans_rate = 79
        self.args.n_tag_sum = 798
        self.args.n_tag_mean = 908
        print(df.columns)

        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        # columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag']
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag']
        if 'test_level_diff' not in df.columns:
            group = df[columns].groupby('userID').apply(
                    lambda r: (
                        r['testId'].values, 
                        r['assessmentItemID'].values,
                        r['KnowledgeTag'].values,
                        r['answerCode'].values,
                        #dev
                        # r['test_level_diff'].values
                    )
                )
        else:
            # columns.append('test_level_diff')
            columns.extend(['test_level_diff','tag_sum','tag_mean','ans_rate'])
            group = df[columns].groupby('userID').apply(
                    lambda r: (
                        r['testId'].values, 
                        r['assessmentItemID'].values,
                        r['KnowledgeTag'].values,
                        r['answerCode'].values,
                        #dev
                        r['test_level_diff'].values,
                        r['tag_mean'].values,
                        r['tag_sum'].values,
                        r['ans_rate'].values
                    )
                )

        return group.values

    def load_train_data(self, file_name):
        # self.train_data = self.load_data_from_file(file_name)
        # self.train_data = self.load_data_from_file_v2(file_name)
        # self.train_data = self.load_data_from_file_v3(file_name)
        self.train_data = self.load_data_from_file_v4(file_name)

    def load_test_data(self, file_name):
        # self.test_data = self.load_data_from_file(file_name, is_train= False)
        # self.test_data = self.load_data_from_file_v2(file_name,is_train=False)
        # self.test_data = self.load_data_from_file_v3(file_name,is_train=False)
        self.test_data = self.load_data_from_file_v4(file_name,is_train=False)

# 범주형, 연속형
class TestDKTDataset(torch.utils.data.Dataset):
    def __init__(self,data, args):
        self.data = data
        self.args = args

    def __getitem__(self,index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        # print(f'row 값 : {len(row)}')
        max_seq_len = self.args.max_seq_len
        global n_cate_cols
        global n_cont_cols
        n_cate_cols = len(self.args.cate_cols)
        n_cont_cols = len(self.args.cont_cols)
        
        cate_cols = [row[i] for i in range(n_cate_cols+1)] # 범주형
        cont_cols = [row[i] for i in range(n_cate_cols+1,len(row))] # 연속형

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다.

        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        tmp_cont_x_list = [] # 임시로 연속형 변수를 저장할 리스트

        for i in range(n_cate_cols+1,len(row)):

            tmp_cont_x = torch.FloatTensor(row[i])
            tmp_cont_x[-1] = 0
            cont_x = torch.FloatTensor(max_seq_len, 1).zero_()
            t = tmp_cont_x[-max_seq_len:]
            # print(f'모양확인 : {t.shape}')
            l = t.shape[0]
            tmp_cont_x = t.view(l,1)
            cont_x[-l:] = tmp_cont_x
            tmp_cont_x_list.append(cont_x)
            # print(cont_x.shape)

        ret_cols = list(cate_cols)
        
        # mask도 columns 목록에 포함시킴
        # cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            # print(f'{i}번째 {col} 값')
            ret_cols[i] = torch.tensor(col)
            # print(f' 츨력 : {ret_cols[i]}')
        mask = torch.tensor(mask)
        ret_cols.append(mask) # 마스크 추가
        # ret_cols.append(cont_x) # 연속형 변수 추가
        ret_cols.extend(tmp_cont_x_list)

        return ret_cols # 반환 범주형, 마스크, 연속형

    def __len__(self):
        return len(self.data)

class MyDKTDataset(torch.utils.data.Dataset):
    def __init__(self,data, args):
        self.data = data
        self.args = args

    def __getitem__(self,index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        # print(f'row 값 : {len(row)}')

        # test, question, tag, correct, test_level_diff, tag_mean, tag_sum, ans_rate
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
            cate_cols[i] = torch.tensor(col)

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

# 범주형, 연속형
def collate_v2(batch):
    # batch 순서 : (범주형),answerCode,(마스크 1개),(연속형)
    # col_n = len(batch[0])
    # for _,i in enumerate(batch):
    #     print(f'{_}번째 batch')
    #     for j in i:
    #         print(type(j),f'길이_연속형 : {j.shape}')

    # print(f'배치 길이 : {len(batch)}')
    # print(f'col_n : {col_n}')
    col_n  = n_cate_cols+ 1 + 1 + n_cont_cols # 앞의 1은 answerCode, 뒤의 1은 mask

    # print(f'col_n : {col_n}')

    # print(len(batch))

    col_list = [[] for _ in range(col_n)]
    #마스크의 길이로 max_seq_len
    max_seq_len = len(batch[0][-1])

    # for i in batch:
    #     print(i)
    #     print('============================================================')
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:

        # 범주형, answerCode, mask
        for i, col in enumerate(row[:(n_cate_cols+1+1)]):
            #앞부분에 마스킹을 넣어주어 sequential하게 interaction들을 학습하게 한다
            
            pre_padded = torch.zeros(max_seq_len)
            if len(col.shape)==2: # 버그 해결용 if 문
                
                l = len(col)
                col = col.permute(1,0)
                # print(col.shape)
                pre_padded[-l:] = col[0]
            else:
                pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)

        # 연속형
        for i, col in enumerate(row[(n_cate_cols+1+1):]):
            col_list[(n_cate_cols+1+1)+i].append(row[(n_cate_cols+1+1)+i])


    for i, _ in enumerate(col_list):
        #stack을 통해 피처 텐서를 이어붙인다(차원축으로) <-> torch.cat
        #각 배치에서 shape(len(feature),len(max_seq_len)) -> shape(len(feature),1,len(max_seq_len)) 
        col_list[i] =torch.stack(col_list[i])
    ret_list = []
    ret_list.extend(col_list[:n_cate_cols+1]) # 범주형 및 answerCode 추가
    
    ret_list.extend(col_list[(n_cate_cols+1+1):]) # 연속형 추가
    ret_list.extend([col_list[n_cate_cols+1]]) # 마스크 추가  
    
    # print(f'차원 : {col_list[4].shape}')
    return tuple(ret_list)

# 범주형만 붙일 때
def collate(batch):
    # batch 순서 : (범주형),(마스크 1개)
    col_n = len(batch[0])
    # print(f'col_n : {col_n}')
    col_list = [[] for _ in range(col_n)]
    #마스크의 길이로 max_seq_len
    max_seq_len = len(batch[0][-1])

    # for i in batch:
    #     print(i)
    #     print('============================================================')
    # batch의 값들을 각 column끼리 그룹화
    # 4+1 = 범주형 4개 + 마스크 1개
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

    old_models = ['lstm, lstmattn, bert, lgbm, lstmroberta, lastquery, saint, lstmalbertattn']
    recent_models = ['TestLSTNConvATTN','gsaint']
    
    if train is not None:

        # trainset = DKTDataset(train, args) # default
        
        trainset = TestDKTDataset(train,args) # 범주형, 연속형
        # trainset = MyDKTDataset(train,args)
        if isinstance(trainset,TestDKTDataset):
            train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate_v2)
        else:
            train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        # valset = DKTDataset(valid, args) # default
        
        valset = TestDKTDataset(valid,args) # 범주형, 연속형
        # valset = MyDKTDataset(valid,args)
        if isinstance(valset,TestDKTDataset):
            valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate_v2)
        else:
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

    del data
    gc.collect()
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
    print(f'Before Augmentation : {len(data)}')
    if args.window == True:
        print("Data Augmentation : Slidding Window")
        data = slidding_window(data, args)
        print(f'After Augmentation : {len(data)}')
    else:
        print(f'No Augmentation')

    

    return data

######################################### Deprecated #################################################################

# 기본 범주형 4 + 연속형 조합을 쓰기 위한 load_data_from_file
def load_data_from_file_v3(self, file_name, is_train=True):
    csv_file_path = os.path.join(self.args.data_dir, file_name)
    print(f'csv_file_path : {csv_file_path}')
    df = pd.read_csv(csv_file_path)
    col_cnt = len(df.columns)
    df = self.__feature_engineering(df)
    df = self.__preprocessing(df, is_train)

    # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용 
    self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
    self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
    self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))

    # user_correct_answer, user_total_answer,user_acc
    print('컬럼 확인')
    print(df.columns)

    df = df.sort_values(by=['userID','Timestamp'], axis=0)
    # columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag'] default 컬럼
    columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag',]
    # user_count 기준으로 other feature를 구성

    # columns.extend(['test_level_diff','tag_sum','tag_mean','ans_rate'])

    if self.args.file_name == 'time_train.csv' or self.args.file_name == 'time_test.csv':
        col_cnt = 6

    columns.extend(list(df.columns[col_cnt:]))
    self.args.n_other_features = [ int(df[i].nunique()) for i in df.columns[col_cnt:]] # 컬럼 순서 꼭 맞출 것!, 추가 컬럼(feature)의 고윳값 수
    
    ret = ['testId','assessmentItemID','KnowledgeTag','answerCode']
    # 도훈님 데이터를 쓸 때
    
    ret.extend(list(df.columns[col_cnt:]))
    print('보낼 최종컬럼 확인')
    print(ret)
    group = df[columns].groupby('userID').apply(
            lambda r: tuple([r[i].values for i in ret])
        )
    
    len(f'group.values->{len(group.values)}')
    return group.values

def load_data_from_file_v2(self, file_name, is_train=True):
    csv_file_path = os.path.join(self.args.data_dir, file_name)
    print(f'csv_file_path : {csv_file_path}')
    df = pd.read_csv(csv_file_path)
    col_cnt = len(df.columns)
    df = self.__feature_engineering(df)
    df = self.__preprocessing(df, is_train)

    # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용 
    self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
    self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
    self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))

    # user_correct_answer, user_total_answer,user_acc
    print('컬럼 확인')
    print(df.columns)

    df = df.sort_values(by=['userID','Timestamp'], axis=0)
    # columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag'] default 컬럼
    columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag',]
    # user_count 기준으로 other feature를 구성

    
    # Optional
    # if self.args.file_name == 'time_train.csv' or self.args.file == 'time_test.csv':
    #     col_cnt = 6

    columns.extend(list(df.columns[col_cnt:]))
    self.args.n_other_features = [ int(df[i].nunique()) for i in df.columns[col_cnt:]] # 컬럼 순서 꼭 맞출 것!, 추가 컬럼(feature)의 고윳값 수
    
    ret = ['testId','assessmentItemID','KnowledgeTag','answerCode']
    # 도훈님 데이터를 쓸 때
    
    ret.extend(list(df.columns[col_cnt:]))
    print('보낼 최종컬럼 확인')
    print(ret)
    group = df[columns].groupby('userID').apply(
            lambda r: tuple([r[i].values for i in ret])
        )
    del df # df는 더이상 쓰이지 않으므로 날림
    len(f'group.values->{len(group.values)}')
    return group.values