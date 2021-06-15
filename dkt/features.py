import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
import numpy as np
import torch



class Features:
    # base_line
    def feature_engineering_01(df):
        #TODO
        #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        df.sort_values(by=['userID','Timestamp'], inplace=True)
        
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

        df = pd.merge(df, correct_t, on=['testId'], how="left")
        df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
        return df

    def feature_engineering_02(df):
        
        return df

    # 종호님 데이터(안쓸 예정)
    def feature_engineering_03(df):
        df = pd.read_csv('/opt/ml/input/data/train_dataset/test_jongho.csv')
        return df

    # 서일님 feature
    def feature_engineering_04(df):

        # testId_mean_sum = df_train_ori.groupby(['testId'])['answerCode'].agg(['mean','sum']).to_dict()
        # assessmentItemID_mean_sum = df_train_ori.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum']).to_dict()
        # KnowledgeTag_mean_sum = df_train_ori.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum']).to_dict()

            # 문항이 중간에 비어있는 경우를 파악 (1,2,3,,5)
        # def assessmentItemID2item(x):
        #     return int(x[-3:]) - 1  # 0 부터 시작하도록 
        # df['item'] = df.assessmentItemID.map(assessmentItemID2item)

        # item_size = df[['assessmentItemID', 'testId']].drop_duplicates().groupby('testId').size()
        # testId2maxlen = item_size.to_dict() # 중복해서 풀이할 놈들을 제거하기 위해

        # item_max = df.groupby('testId').item.max()
        # print(len(item_max[item_max + 1 != item_size]), '개의 시험지가 중간 문항이 빈다. item_order가 올바른 순서') # item_max는 0부터 시작하니까 + 1
        # shit_index = item_max[item_max +1 != item_size].index
        # shit_df = df.loc[df.testId.isin(shit_index),['assessmentItemID', 'testId']].drop_duplicates().sort_values('assessmentItemID')      
        # shit_df_group = shit_df.groupby('testId')

        # shitItemID2item = {}
        # for key in shit_df_group.groups:
        #     for i, (k,_) in enumerate(shit_df_group.get_group(key).values):
        #         shitItemID2item[k] = i
            
        # def assessmentItemID2item_order(x):
        #     if x in shitItemID2item:
        #         return int(shitItemID2item[x])
        #     return int(x[-3:]) - 1  # 0 부터 시작하도록 
        # df['item_order'] =  df.assessmentItemID.map(assessmentItemID2item_order)



        # #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        # df.sort_values(by=['userID','Timestamp'], inplace=True)
        
        # # 유저가 푼 시험지에 대해, 유저의 전체 정답/풀이횟수/정답률 계산 (3번 풀었으면 3배)
        # df_group = df.groupby(['userID','testId'])['answerCode']
        # df['user_total_correct_cnt'] = df_group.transform(lambda x: x.cumsum().shift(1))
        # df['user_total_ans_cnt'] = df_group.cumcount()
        # df['user_total_acc'] = df['user_total_correct_cnt'] / df['user_total_ans_cnt']

        # # 유저가 푼 시험지에 대해, 유저의 풀이 순서 계산 (시험지를 반복해서 풀었어도, 누적되지 않음)
        # # 특정 시험지를 얼마나 반복하여 풀었는지 계산 ( 2번 풀었다면, retest == 1)
        # df['test_size'] = df.testId.map(testId2maxlen)
        # df['retest'] = df['user_total_ans_cnt'] // df['test_size']
        # df['user_test_ans_cnt'] = df['user_total_ans_cnt'] % df['test_size']

        # # 각 시험지 당 유저의 정확도를 계산
        # df['user_test_correct_cnt'] = df.groupby(['userID','testId','retest'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
        # df['user_acc'] = df['user_test_correct_cnt']/df['user_test_ans_cnt']

        # # 본 피처는 train에서 얻어진 값을 그대로 유지합니다.
        # df["test_mean"] = df.testId.map(testId_mean_sum['mean'])
        # df['test_sum'] = df.testId.map(testId_mean_sum['sum'])
        # df["ItemID_mean"] = df.assessmentItemID.map(assessmentItemID_mean_sum['mean'])
        # df['ItemID_sum'] = df.assessmentItemID.map(assessmentItemID_mean_sum['sum'])
        # df["tag_mean"] = df.KnowledgeTag.map(KnowledgeTag_mean_sum['mean'])
        # df['tag_sum'] = df.KnowledgeTag.map(KnowledgeTag_mean_sum['sum'])

        return df
    def feature_engineering_05(df):

        #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        df.sort_values(by=['userID','Timestamp'], inplace=True)
        
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

        df = pd.merge(df, correct_t, on=['testId'], how="left")
        df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
        
        
        # last_Tag ans rate
        last_tag_ans_rate = self.last_tag_rate(df)
        df = pd.merge(df, last_tag_ans_rate, on=['userID'], how='left')
        # df=df.drop(columns=['_'])
        print(f'df 살펴보기')
        print('정답률 : '+str(df['ans_rate'].nunique()))
        print('태그개수 고윳값 : '+str(df['tag_sum'].nunique()))
        print('태그평균 고윳값 : '+str(df['tag_mean'].nunique()))
        print(f'태그합 최댓값 : '+str(df['tag_sum'].max()))

        _max_sum = df['tag_sum'].max()
        _max_mean = df['tag_mean'].max()

        df['tag_sum'] = df['tag_sum'].apply(lambda x : x/(10*len(str(_max_sum))))
        df['tag_mean'] = df['tag_mean'].apply(lambda x : x/(10*len(str(_max_mean))))

        return df[['ans_rate','tag_sum','tag_mean']]

    # last Tag에 대한 정답률 함수 생성
    def last_tag_rate(df):
        df = df.copy()
        
        last_tag = df.groupby(['userID']).tail(1)
        last_tag_1 = last_tag.loc[:, ['userID','KnowledgeTag']]
        
        user_tag_ans = last_tag_1.merge(df, on=['userID','KnowledgeTag'], how='left')
        
        user_tag_ans['count'] = user_tag_ans.groupby(['userID','KnowledgeTag'])['KnowledgeTag'].transform('count')
        user_tag_ans['tag_cnt'] = user_tag_ans['count']-1
        user_tag_ans['ans_cnt'] = user_tag_ans.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        
        user_tag_ans_rate = user_tag_ans.groupby('userID').tail(1)
        user_tag_ans_rate['ans_cnt'] = user_tag_ans_rate['ans_cnt'].fillna(0)
        user_tag_ans_rate['ans_rate'] = round(user_tag_ans_rate['ans_cnt']/ user_tag_ans_rate['tag_cnt'],2)
        user_tag_ans_rate['ans_rate'] = user_tag_ans_rate['ans_rate'].fillna(0.00)
        
        return user_tag_ans_rate.loc[:,['userID','ans_rate']]

    # 新 채원님 Feature
    def feature_engineering_06(df):

        def test_rate(df):
            test_df = df.groupby(['testId'])['testId'].count().reset_index(name='test_cnt')
            test_ans_df = df.groupby(['testId'])['answerCode'].sum().reset_index(name='test_ans_cnt')
            test_df = test_df.merge(test_ans_df, on ='testId', how='left')
            test_df['test_rate'] = round(test_df['test_ans_cnt'] / test_df['test_cnt'], 2)
            test_df['test_rate'] = test_df['test_rate'].fillna(0.00)
            return test_df.loc[:,['testId','test_rate']]

        def que_rate(df):
            que_df = df.groupby(['assessmentItemID'])['assessmentItemID'].count().reset_index(name='que_cnt')
            que_ans_df = df.groupby(['assessmentItemID'])['answerCode'].sum().reset_index(name='que_ans_cnt')
            que_df = que_df.merge(que_ans_df, on ='assessmentItemID', how='left')
            que_df['que_rate'] = round(que_df['que_ans_cnt'] / que_df['que_cnt'], 2)
            que_df['que_rate'] = que_df['que_rate'].fillna(0.00)
            return que_df.loc[:,['assessmentItemID','que_rate']]

        def user_test_rate(df):
            df['index'] = df.index
            u_test_cnt = df.groupby(['userID','testId'])['testId'].cumcount().reset_index(name='u_test_cnt')
            user_test = df.merge(u_test_cnt, on='index', how='left')
            u_test_cnt= user_test.groupby(['userID','testId'])['answerCode'].transform(lambda x: x.cumsum().shift(1)).reset_index(name='test_ans_cnt')
            user_test_ans_sum =  user_test.merge(u_test_cnt, on='index', how='left')
            user_test_ans_sum['test_ans_cnt'] = user_test_ans_sum['test_ans_cnt'].fillna(0.0)
            
            def rating_1(user_test_ans_sum):
                if user_test_ans_sum['u_test_cnt'] == 0:
                    return 0.50
                else :    
                    return round(user_test_ans_sum['test_ans_cnt']/user_test_ans_sum['u_test_cnt'],2)
                
            user_test_ans_sum['test_ans_rate'] = user_test_ans_sum.apply(rating_1, axis=1)
            
            return user_test_ans_sum.loc[:,['index','u_test_cnt','test_ans_cnt','test_ans_rate']]
        
        # user별 Tag에 대한 정답률 sequential하게 적용
        def ut_ans_rate(df) :
            df['index'] = df.index

            u_t_cnt = df.groupby(['userID','KnowledgeTag'])['KnowledgeTag'].cumcount().reset_index(name='u_tag_cnt')
            user_tag =  df.merge(u_t_cnt, on='index', how='left')
            #user_tag['tag_cnt_1'] = user_tag['u_tag_cnt']+1

            u_t_cnt= user_tag.groupby(['userID','KnowledgeTag'])['answerCode'].transform(lambda x: x.cumsum().shift(1)).reset_index(name='ans_cnt')
            user_tag_ans_sum =  user_tag.merge(u_t_cnt, on='index', how='left')
            user_tag_ans_sum['ans_cnt'] = user_tag_ans_sum['ans_cnt'].fillna(0.0)

            def rating(user_tag_ans_sum):
                if user_tag_ans_sum['u_tag_cnt'] == 0:
                    return 0.50
                else :    
                    return round(user_tag_ans_sum['ans_cnt']/user_tag_ans_sum['u_tag_cnt'],2)
            user_tag_ans_sum['tag_ans_rate'] = user_tag_ans_sum.apply(rating, axis=1)
            
            return user_tag_ans_sum.loc[:,['index','u_tag_cnt','ans_cnt','tag_ans_rate']]

        df['index'] = df.index

        #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        df.sort_values(by=['userID','Timestamp'], inplace=True)
        
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

        df = pd.merge(df, correct_t, on=['testId'], how="left")
        df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
        
        
    #     # last_Tag ans rate
    #     last_tag_ans_rate = last_tag_rate(df)
    #     df = pd.merge(df, last_tag_ans_rate, on=['userID'], how='left')
        
        # test별 정답률
        all_test_rate = test_rate(df)
        df = pd.merge(df, all_test_rate, on='testId', how='left')
        

            ########## 이부분 추가 ##########
        # 문제별 정답률
        all_que_rate = que_rate(df)
        df = pd.merge(df, all_que_rate, on='assessmentItemID', how='left')
        ################################

        
        # user_test_ans_rate
        user_test_ans_rate = user_test_rate(df)
        df = pd.merge(df, user_test_ans_rate, on='index', how='left')
        

        # user_tag_seq_ans_rate     
        user_tag_ans_rate = ut_ans_rate(df)
        df = pd.merge(df, user_tag_ans_rate, on='index', how='left')

        for i in df.columns:
            print(f'컬럼 {i}의 고윳값 개수 : {str(df[i].nunique())}')

        # tmp_list = ['user_correct_answer','user_total_answer'] # 소숫점화 시킬 대상
        # for i in tmp_list:
        #     m = df[i].max()
        #     print(f'최댓값 : {m}')
        #     df[i]=df[i].fillna(0).apply(lambda x : x/(10**len(str(int(m)))))
        df['user_acc'] = df['user_acc'].fillna(0)
        df['user_correct_answer'] = df['user_correct_answer'].fillna(0)
        df['user_total_answer'] = df['user_total_answer'].fillna(0)
        return df[['user_acc','user_correct_answer', 'user_total_answer']]

    # 그룹병 정답 누적합
    def feature_engineering_07(df):
        df['test_user_correct_answer'] = df.groupby(['userID','testId'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
        m = df['test_user_correct_answer'].max()
        df['test_user_correct_answer'] = df['test_user_correct_answer'].fillna(0).apply(lambda x : x/(10**len(str(int(m)))))
        return df

    # 그룹별 푼 문제 개수 누적 합
    def feature_engineering_08(df):
        df['test_user_total_answer'] = df.groupby(['userID','testId'])['answerCode'].cumcount()
        m = df['test_user_total_answer'].max()

        df['test_user_total_answer'] = df['test_user_total_answer'].fillna(0).apply(lambda x : x/(10**len(str(int(m)))))
        return df

    # 누적정답률
    def feature_engineering_09(df):
        df['test_user_correct_answer'] = df.groupby(['userID','testId'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
        df['test_user_total_answer'] = df.groupby(['userID','testId'])['answerCode'].cumcount()
        df['test_user_acc'] = df['test_user_correct_answer']/df['test_user_total_answer']
        
        return df.fillna(0.5)

    # 정답 누적합
    def feature_engineering_10(df):
        df['tag_user_correct_answer_tag'] = df.groupby(['userID','KnowledgeTag'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
        m = df['tag_user_correct_answer_tag'].max()
        df['tag_user_correct_answer_tag'] = df['tag_user_correct_answer_tag'].fillna(0).apply(lambda x : x/(10**len(str(int(m)))))
        return df

    # 푼 문제 개수 누적 합
    def feature_engineering_11(df):
        df['tag_user_total_answer_tag'] = df.groupby(['userID','KnowledgeTag'])['answerCode'].cumcount()
        m = df['tag_user_correct_answer_tag'].max()
        df['tag_user_total_answer_tag'] = df['tag_user_total_answer_tag'].fillna(0).apply(lambda x : x/(10**len(str(int(m)))))
        return df

    # 누적 정답률
    def feature_engineering_12(df):
        df['tag_user_correct_answer_tag'] = df.groupby(['userID','KnowledgeTag'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
        df['tag_user_total_answer_tag'] = df.groupby(['userID','KnowledgeTag'])['answerCode'].cumcount()
        df['tag_user_acc_tag'] = df['tag_user_correct_answer_tag']/df['tag_user_total_answer_tag']
        
        return df.fillna(0.5)

    # 문제를 푼 시간(by 도훈님), 유저별 (현재 time_stamp - 과거 time_stamp)
    def feature_engineering_13(df):
        df = pd.read_csv('/opt/ml/input/data/train_dataset/time_train.csv')
        df=df.sort_values(by=["userID","sec_time"],ascending=True)
        df['time_diff'] = df.groupby('userID')['sec_time'].apply(lambda x: (x-x.shift(1))).fillna(0).apply(lambda x : min(int(x),3600*24*3))
        _max = int(df['time_diff'].max())
        df['time_diff'] = df['time_diff'].apply(lambda x:x/(10**(len(f'{max}'))))
        _max = int(df['solve_time'].max())
        df['solve_time'] = df['solve_time'].apply(lambda x:x/(10**(len(f'{max}'))))

        return df

    # 풀이시간대 별 정답률
    def hour_rate(df):
        test_df = df.groupby(['hour'])['hour'].count().reset_index(name='hour_cnt')
        test_ans_df = df.groupby(['hour'])['answerCode'].sum().reset_index(name='hour_ans_cnt')
        test_df = test_df.merge(test_ans_df, on ='hour', how='left')
        test_df['hour_rate'] = round(test_df['hour_ans_cnt'] / test_df['hour_cnt'], 2)
        return test_df

    # 학년 발굴
    def feature_engineering_14(df):
        df['grade'] = df['testId'].apply(lambda x: x[2]).astype(int)
        return df

    def feature_engineering_15(df):
        from sklearn.preprocessing import LabelEncoder, QuantileTransformer
        # 수준
        df['test_level'] = df['testId'].str[2]

        # 문제 푼 시간
        df['tmp_index'] = df.index
        tmp_df = df[['userID', 'testId', 'Timestamp', 'tmp_index']].shift(1)
        tmp_df['tmp_index'] += 1
        tmp_df = tmp_df.rename(columns={'Timestamp':'prior_timestamp'})
        df = df.merge(tmp_df, how='left', on=['userID', 'testId', 'tmp_index'])
        df['solve_time'] = (df.Timestamp - df.prior_timestamp).dt.seconds

        upper_bound = df['solve_time'].quantile(0.98) # outlier 설정
        median = df[df['solve_time'] <= upper_bound]['solve_time'].median() 
        df.loc[df['solve_time'] > upper_bound, 'solve_time'] = median 
        df['solve_time'] = df['solve_time'].fillna(median) # 빈값을 중앙값으로 채우기

        # 수치형 transform
        df['solve_time'] = np.log1p(df['solve_time']) #
        df['solve_time'] = QuantileTransformer(output_distribution='normal').fit_transform(df.solve_time.values.reshape(-1,1)).reshape(-1) 

        # 문제 평균 소모시간
        assess_time = df.groupby('assessmentItemID').solve_time.mean()
        assess_time.name = 'mean_elapsed'
        df = df.merge(assess_time, how='left', on=['assessmentItemID'])

        # 테스트 평균 소모시간
        test_time = df.groupby('testId').solve_time.mean()
        test_time.name = 'test_time'
        df = df.merge(test_time, how='left', on=['testId'])

        # 대분류별 평균 소모시간
        grade_time = df.groupby('test_level').solve_time.mean()
        grade_time.name = 'grade_time'
        df = df.merge(grade_time, how='left', on=['test_level'])

        # user&태그별 누적 정답횟수
        df['tag_cumAnswer'] = df.groupby(['userID', 'KnowledgeTag']).answerCode.cumsum() - df['answerCode']
        df['tag_cumAnswer'] = np.log1p(df['tag_cumAnswer'])


        # 비율 계산
        def percentile(s):
            return np.sum(s) / len(s)
            
        # 수준 카테고리를 int로 매핑
        df['test_level_int'] = df['testId'].str[2].astype(int)

        # 마지막 row, answerCode -1 는 제외한 후 평균을 계산한다.
        minus_answerCode = df[df['answerCode'] == -1].index
        cal_df = df.drop(index=minus_answerCode)

        # 수준별 정답률
        stu_groupby = cal_df.groupby('test_level_int').agg({
            'assessmentItemID': 'count',
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # tag별 정답률
        stu_tag_groupby = cal_df.groupby(['KnowledgeTag']).agg({
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # 시험지별 정답률
        stu_test_groupby = cal_df.groupby(['testId']).agg({
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # 문항별 정답률
        stu_assessment_groupby = cal_df.groupby(['assessmentItemID']).agg({
            'answerCode': percentile
        }).rename(columns = {'assessmentItemID' : 'assessment_count', 'answerCode' : 'answer_rate'})

        # 수준&tag별 정답률
        level_tag_groupby = df.groupby(['test_level_int', 'KnowledgeTag']).agg({
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # 수준&시험지별 정답률
        level_test_groupby = df.groupby(['test_level_int', 'testId']).agg({
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # 수준&문항별 정답률 + 문제count
        level_assessment_groupby = df.groupby(['test_level_int', 'assessmentItemID']).agg({
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # 정답 - 수준별 정답률 
        df = df.merge(stu_groupby.reset_index()[['test_level_int', 'answer_rate']], on=['test_level_int'])
        df = df.rename(columns={'answer_rate':'answer_acc'})

        # 정답 - 태그별 정답률
        df = df.merge(stu_tag_groupby.reset_index()[['answer_rate', 'KnowledgeTag']], on=['KnowledgeTag'])
        df = df.rename(columns={'answer_rate':'tag_acc'})

        # 정답 - 시험별 정답률
        df = df.merge(stu_test_groupby.reset_index()[['answer_rate', 'testId']], on=['testId'])
        df = df.rename(columns={'answer_rate':'test_acc'})

        # 정답 - 문항별 정답률
        df = df.merge(stu_assessment_groupby.reset_index()[['answer_rate', 'assessmentItemID']], on=['assessmentItemID'])
        df = df.rename(columns={'answer_rate':'assess_acc'})

        # 정답 - 수준 & 태그별 정답률
        df = df.merge(level_tag_groupby.reset_index()[['answer_rate', 'KnowledgeTag', 'test_level_int']], on=['test_level_int', 'KnowledgeTag'])
        df = df.rename(columns={'answer_rate':'level_tag_acc'})

        # 정답 - 수준 & 시험별 정답률
        df = df.merge(level_test_groupby.reset_index()[['answer_rate', 'testId', 'test_level_int']], on=['test_level_int', 'testId'])
        df = df.rename(columns={'answer_rate':'level_test_acc'})

        # 정답 - 수준 & 문항별 정답률
        df = df.merge(level_assessment_groupby.reset_index()[['answer_rate', 'assessmentItemID', 'test_level_int']], on=['test_level_int', 'assessmentItemID'])
        df = df.rename(columns={'answer_rate':'level_assess_acc'})

        return df

    # test_diff
    def feature_engineering_16(df):
        def percentile(s):
            return np.sum(s) / len(s)
        def fun(x):
            if x > 0.8 :
                return 5
            elif x <=0.8 and x >0.6 :
                return 4
            elif x <=0.6 and x >0.4 :
                return 3
            elif x <=0.4 and x >0.2 :
                return 2
            elif x <=0.2  :
                return 1
        df1=df.groupby('userID').agg({'answerCode': percentile})
        df1.columns = ["test_rate"]
        df1.reset_index(inplace = True)
        df2 = df1.copy()
        df2['level']=pd.Series([fun(x) for x in df2['test_rate']])
        del df2['test_rate']
        df = pd.merge(df,df2,on='userID')
        df3 =df.groupby(['assessmentItemID','level']).agg({'answerCode':['sum','count']})
        df3.reset_index(inplace=True)
        df3.columns = ['assessmentItemID','point','sum','count']
        df4 = df3.copy()
        df4 = df4.sort_values(by="point", ascending=True).groupby("assessmentItemID").head(1)
        df4.drop(["point", "count"], axis=1,inplace = True)
    #     del df4['point']
    #     del df4['count']
        df4.columns = ['assessmentItemID','answer_min_count']
        df5 = df3.copy()
        df5 = df5.sort_values(by="point", ascending=False).groupby("assessmentItemID").head(1)
        df5.drop(["point", "count"], axis=1,inplace = True)
    #     del df5['point']
    #     del df5['count']
        df5.columns = ['assessmentItemID','answer_max_count']
        df = pd.merge(df,df4,on='assessmentItemID')
        df = pd.merge(df,df5,on='assessmentItemID')
        df6 = df.groupby('assessmentItemID').agg({'userID':'count'})
        df6.columns = ['user_count']
        df = pd.merge(df,df6,on='assessmentItemID')
        df['test_level_diff_itemID']=pd.Series((df['answer_max_count']-df['answer_min_count'])/(df['user_count']/2))
        df.drop(["answer_max_count", "answer_min_count",'user_count'], axis=1,inplace = True)
    ###################################################################################################################################
        df7 =df.groupby(['KnowledgeTag','level']).agg({'answerCode':['sum','count']})
        df7.reset_index(inplace=True)
        df7.columns = ['KnowledgeTag','point','sum','count']
        df8 = df7.copy()
        df8= df8.sort_values(by="point", ascending=True).groupby("KnowledgeTag").head(1)
        del df8['point']
        del df8['count']
        df8.columns = ['KnowledgeTag','answer_min_count']
        df9 = df7.copy()
        df9 = df9.sort_values(by="point", ascending=False).groupby("KnowledgeTag").head(1)
        del df9['point']
        del df9['count']
        df9.columns = ['KnowledgeTag','answer_max_count']
        df = pd.merge(df,df8,on='KnowledgeTag')
        df = pd.merge(df,df9,on='KnowledgeTag')
        df10 = df.groupby('KnowledgeTag').agg({'userID':'count'})
        df10.columns = ['user_count']
        df = pd.merge(df,df10,on='KnowledgeTag')
        df['test_level_diff_KnowledgeTag']=pd.Series((df['answer_max_count']-df['answer_min_count'])/(df['user_count']/2))
        df.drop(["answer_max_count", "answer_min_count",'user_count'], axis=1,inplace = True)
    ###################################################################################################################################
        df11 =df.groupby(['testId','level']).agg({'answerCode':['sum','count']})
        df11.reset_index(inplace=True)
        df11.columns = ['testId','point','sum','count']
        df12 = df11.copy()
        df12= df12.sort_values(by="point", ascending=True).groupby("testId").head(1)
        del df12['point']
        del df12['count']
        df12.columns = ['testId','answer_min_count']
        df13 = df11.copy()
        df13 = df13.sort_values(by="point", ascending=False).groupby("testId").head(1)
        del df13['point']
        del df13['count']
        df13.columns = ['testId','answer_max_count']
        df = pd.merge(df,df12,on='testId')
        df = pd.merge(df,df13,on='testId')
        df14 = df.groupby('testId').agg({'userID':'count'})
        df14.columns = ['user_count']
        df = pd.merge(df,df14,on='testId')
        df['test_level_diff_testId']=pd.Series((df['answer_max_count']-df['answer_min_count'])/(df['user_count']/2))
        df.drop(["answer_max_count", "answer_min_count",'user_count'], axis=1,inplace = True)
        return df


    # def month_augmentation(df):
    #     # import pandas as pd
    #     df['Timestamp'] = pd.to_datetime(df['Timestamp'].values)
    #     df['month'] = df['Timestamp'].dt.month
    #     df['user_id'] = df['userID'].map(str)+'0'+df['month'].map(str)
    #     df['user_id'] = df['user_id'].astype('int32')
    #     del df['userID']
    #     df.rename(columns = {'user_id' : 'userID'}, inplace = True)
    #     return df