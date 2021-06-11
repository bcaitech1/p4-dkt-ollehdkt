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


def make_sharing_feature(args):
    """[use train+test(except last row) get pre_processed feature]

    Args:
        args ([conf.yaml]): [args를 전달받는다]
    """
    
    csv_file_path = os.path.join(args.data_dir, args.file_name)
    df = pd.read_csv(csv_file_path)#, nrows=100000)
    csv_file_path = os.path.join(args.data_dir, args.test_file_name)
    tdf = pd.read_csv(csv_file_path)#, nrows=100000)

    if args.use_distance:
        df['distance']=np.load('/opt/ml/np_train_tag_distance_arr.npy')
        tdf['distance']=np.load('/opt/ml/np_test_tag_distance_arr.npy')
        
    tdf=tdf[tdf['userID']==tdf['userID'].shift(-1)]
    df=pd.concat([df,tdf],ignore_index=True)
    return df


def get_sharing_feature(args):
    if args.make_sharing_feature:
        df=make_sharing_feature(args)
    else :
        csv_file_path = os.path.join(args.data_dir, args.file_name)
        df = pd.read_csv(csv_file_path)#, nrows=100000)
    # trian에서 각 문제 평균 뽑기
    testId_mean_sum = df.groupby(['testId'])['answerCode'].agg(['mean','sum']).to_dict()
    assessmentItemID_mean_sum = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum']).to_dict()
    KnowledgeTag_mean_sum = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum']).to_dict()
    
    # 시간 피처
    testId_time_agg = df.groupby(['testId'])['solve_time'].agg(['mean','std','skew']).to_dict()
    assessment_time_agg=df.groupby(['assessmentItemID'])['solve_time'].agg(['mean','std','skew']).to_dict()
    KnowledgeTag_time_agg = df.groupby(['KnowledgeTag'])['solve_time'].agg(['mean','std','skew']).to_dict()

    #해당 문제를 맞은사람의 평균시간과 틀린사람의 평균시간
    a_t_rate_df=df.groupby(['assessmentItemID','answerCode']).agg({'solve_time':'mean'}).reset_index(drop=False)
    assess_time_corNwrong_agg=a_t_rate_df.groupby('assessmentItemID')['solve_time'].agg(['first','last']).to_dict()
    
    return testId_mean_sum, assessmentItemID_mean_sum, KnowledgeTag_mean_sum,testId_time_agg,assessment_time_agg,KnowledgeTag_time_agg,assess_time_corNwrong_agg

def make_lgbm_feature(args, df,is_train=True):
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
    # 유저가 푼 시험지에 대해, 유저의 전체 정답/풀이횟수/정답률 계산 (3번 풀었으면 3배)
    df_group = df.groupby(['userID','testId'])['answerCode']
    df['user_total_correct_cnt'] = df_group.transform(lambda x: x.cumsum().shift(1))
    df['user_total_ans_cnt'] = df_group.cumcount()
    df['user_total_acc'] = df['user_total_correct_cnt'] / df['user_total_ans_cnt']

    # 유저가 푼 시험지에 대해, 유저의 풀이 순서 계산 (시험지를 반복해서 풀었어도, 누적되지 않음)
    # 특정 시험지를 얼마나 반복하여 풀었는지 계산 ( 2번 풀었다면, retest == 1)
    df['test_size'] = df.testId.map(testId2maxlen)
    df['retest'] = df['user_total_ans_cnt'] // df['test_size']
    df['user_test_ans_cnt'] = df['user_total_ans_cnt'] % df['test_size']

    # 각 시험지 당 유저의 정확도를 계산
    df['user_test_correct_cnt'] = df.groupby(['userID','testId','retest'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_acc'] = df['user_test_correct_cnt']/df['user_test_ans_cnt']

    ###서일님 피처

    #sequential feature in here
    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    # df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    # df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    # df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    #학생의 학년을 정하고 푼 문제지의 학년합을 구해본다
    df['test_level']=df['assessmentItemID'].apply(lambda x:int(x[2]))
    #문제번호
    df['problem_number']=df['assessmentItemID'].apply(lambda x:int(x[-3:]))
    #시간
    # df['year_month']=pd.to_datetime(df['Timestamp'], format="").dt.strftime('%Y-%m')
    # print(time.dt.strftime('%Y-%m-%d'))
    df['test_tag_cumsum']=df.groupby(['userID','testId']).agg({'solve_time':'cumsum'})
    # non-sequential feature in here
    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    group_list=['userID']
    
    #문제 태그로 groupby했을 때 적용할 함수들
    # tag_agg_dict={
    #     # 'answerCode': ['count','mean', 'sum'], #태그개수,태그별 정답률, 해당 태그를 맞춘 개수
    #     # 'userID' :['nunique'], #해당 태그를 풀이한 유저의 수(인기도), 
    #     # 'assessmentItemID':['nunique'], #해당 태그가 얼마나 여러 문제번호에 분포돼있는지, 왜도(문제지의 어느부분인지)
    #     'solve_time' :['mean','std','skew'],
    # }
    
    #시험지번호로 groupby했을 때 적용할 함수들
    # test_agg_dict={
    #     'solve_time' :['mean','std','skew'],
    #     # 'answerCode': ['count','mean', 'sum'], #시험지별 제출개수 ,시험지별 정답률, 해당 시험지를 풀이하여 맞은 개수
    #     # 'userID' :['nunique'], #해당 시험지를 풀이한 유저의 수(인기도), 
    #     # 'assessmentItemID':['nunique'], #해당 시험지에 문제가 얼마나 분포돼있는지, 왜도(문제지의 어느부분인지)
        
    # }
    
    #사용자별로 groupby했을 때 적용할 함수들, answercode 관련 column은 위에서 이미 정의함
    uid_agg_dict={
        # 'assessmentItemID':['nunique'], #얼마나 많은 종류의 문제를 풀었는지, 왜도(문제지의 어느부분인지)
        # 'problem_number':['skew'],
        # 'year_month':[lambda x:x.value_counts().index[0]],
        # 'Timestamp':['first'],
        # 'test_level':['mean', 'sum','std'],
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

    #agg취한 값들이 feature가 된다
    delete_feats=['userID','assessmentItemID','testId','answerCode','Timestamp','sec_time']
    features = df.drop(columns=delete_feats).columns
    
    df.isnull().sum()
    df = df.fillna(0)
    
    
    return lgbm_feature_preprocessing(df,features, do_imputing=True)

def lgbm_split_data(data,ratio,seed=42):
    random.seed(seed)

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

def lgbm_oof_split_data_withidx(args,data):
    random.seed(42)
    n_fold=args.n_fold
    
    users = list(zip(data['userID'].value_counts().index, data['userID'].value_counts()))
    random.shuffle(users)
    
    user_id_dict=defaultdict(list)
    user_count_dict=defaultdict(int)
    
    for idx, (user_id, count) in enumerate(users):
        f_num=idx%n_fold
        user_count_dict[f_num] += count
        user_id_dict[f_num].append(user_id)

    return user_id_dict,user_count_dict

def get_fold_data(idx,data,user_id_dict,user_count_dict):
    
    train = data[data['userID'].isin(user_id_dict[idx]) == False]
    test = data[data['userID'].isin(user_id_dict[idx])]

    #test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test

def lgbm_make_test_data(data):
    data = data.drop(['answerCode'], axis=1)
    return data[data['userID'] != data['userID'].shift(-1)]

def lgbm_train(args,train_data,valid_data):
    # 사용할 Feature 설정
    delete_feats=['userID','assessmentItemID','testId','answerCode','Timestamp','sec_time']

    FEATS = list(set(train_data.columns)-set(delete_feats))
    print(f'{len(FEATS)}개의 피처를 사용합니다')
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
        
                args.lgbm.model_params,
                lgb_train,
                valid_sets=[lgb_train, lgb_test],
                verbose_eval=args.model_params.verbose_eval,
                num_boost_round=args.model_params.num_boost_round,
                early_stopping_rounds=args.model_params.early_stopping_rounds,
            )

    preds = model.predict(valid_data[FEATS])
    acc = accuracy_score(y_test, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_test, preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')
    
    ax = lgb.plot_importance(model,figsize=(12,7))
    #Feature Importance 저장
    new_output_path=f'{args.output_dir}{args.task_name}'
    write_path = os.path.join(new_output_path, "feature_importance.png")
    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path) 
    ax.figure.savefig(write_path,bbox_inches='tight', pad_inches=0.5)

    return model,auc,acc
    
def lgbm_inference(args,model, test_data):
    
    delete_feats=['userID','assessmentItemID','testId','answerCode','Timestamp','sec_time']
    FEATS = list(set(test_data.columns)-set(delete_feats))
    
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
    
def lgbm_feature_preprocessing(train,features, do_imputing=True):
    x_tr = train.copy()
    
    # 범주형 피처 이름을 저장할 변수
    cate_cols = []

    # 레이블 인코딩
    for f in features:
        if x_tr[f].dtype.name == 'object': # 데이터 타입이 object(str)이면 레이블 인코딩
            cate_cols.append(f)
            le = LabelEncoder()
            # train 데이터를 레이블 인코딩 함수에 fit
            le.fit(list(x_tr[f].values))
            
            # train 데이터 레이블 인코딩 변환 수행
            x_tr[f] = le.transform(list(x_tr[f].values))
            

    print('categorical feature:', cate_cols)

    if do_imputing:
        # 중위값으로 결측치 채우기
        imputer = SimpleImputer(strategy='median')
        x_tr[features] = imputer.fit_transform(x_tr[features])
    
    return x_tr

def make_lgb_user_oof_prediction(args, train, test, features, categorical_features='auto', model_params=None, folds=None):
    user_id_dict,user_count_dict=lgbm_oof_split_data_withidx(args,train)

    test = test[test['userID'] != test['userID'].shift(-1)]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    acc=0
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    for fold in range(folds):
        # train index, validation index로 train 데이터를 나눔
        train_set,valid_set=get_fold_data(fold,train,user_id_dict,user_count_dict)
        x_tr, x_val = train_set[features], valid_set[features]
        y_tr, y_val = train_set['answerCode'], valid_set['answerCode']
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # LightGBM 데이터셋 선언
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_val, label=y_val)
        
        # LightGBM 모델 훈련
        clf = lgb.train(
            model_params,
            dtrain,
            valid_sets=[dtrain, dvalid], # Validation 성능을 측정할 수 있도록 설정
            categorical_feature=categorical_features,
            verbose_eval=args.lgbm.verbose_eval,
            num_boost_round=args.lgbm.num_boost_round,
            early_stopping_rounds=args.lgbm.early_stopping_rounds,
        )

        # Validation 데이터 예측
        val_preds = clf.predict(x_val)
        
        # Validation index에 예측값 저장 
        # y_oof[fold] = val_preds
        
        # 폴드별 Validation 스코어 측정
        fold_acc=accuracy_score(y_val, list(map(round,val_preds)))
        
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)} | ACC: {fold_acc}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        acc+=fold_acc / folds
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(x_test) / folds
        
        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = clf.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"Mean ACC = {acc}") # 폴드별 Validation 스코어 출력
    # print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
    
    # 폴드별 피처 중요도 평균값 계산해서 저장 
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    #add fig to mlflow
    n=40
    color='blue'
    figsize=(12,8)
    
    fi = fi.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # 피처 중요도 정규화 및 누적 중요도 계산
    fi['importance_normalized'] = fi['importance'] / fi['importance'].sum()
    fi['cumulative_importance'] = np.cumsum(fi['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    plt.style.use('fivethirtyeight')
    # 피처 중요도 순으로 n개까지 바플롯으로 그리기
    fi.loc[:n, :].plot.barh(y='importance_normalized', 
                            x='feature', color=color, 
                            edgecolor='k', figsize=figsize,
                            legend=False)

    plt.xlabel('Normalized Importance', size=18); plt.ylabel(''); 
    plt.title(f'Top {n} Most Important Features', size=18)
    plt.gca().invert_yaxis()

    new_output_path=f'{args.output_dir}{args.task_name}'
    write_path = os.path.join(new_output_path, "feature_importance.png")
    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path) 
    plt.savefig(write_path,bbox_inches='tight', pad_inches=0.5)
    plt.close()   
    
    return y_oof, test_preds, fi , score, acc


def make_lgb_oof_prediction(args,train, test, features, categorical_features='auto', model_params=None, folds=None):
    x_train = train[features]
    y =train['answerCode']
    test = test[test['userID'] != test['userID'].shift(-1)]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # LightGBM 데이터셋 선언
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_val, label=y_val)
        
        # LightGBM 모델 훈련
        clf = lgb.train(
            model_params,
            dtrain,
            valid_sets=[dtrain, dvalid], # Validation 성능을 측정할 수 있도록 설정
            categorical_feature=categorical_features,
            verbose_eval=args.lgbm.verbose_eval,
            num_boost_round=args.lgbm.num_boost_round,
            early_stopping_rounds=args.lgbm.early_stopping_rounds,
        )

        # Validation 데이터 예측
        val_preds = clf.predict(x_val)
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(x_test) / folds
        
        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = clf.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
    
    # 폴드별 피처 중요도 평균값 계산해서 저장 
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    #add fig to mlflow
    n=40
    color='blue'
    figsize=(12,8)
    
    fi = fi.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # 피처 중요도 정규화 및 누적 중요도 계산
    fi['importance_normalized'] = fi['importance'] / fi['importance'].sum()
    fi['cumulative_importance'] = np.cumsum(fi['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    plt.style.use('fivethirtyeight')
    # 피처 중요도 순으로 n개까지 바플롯으로 그리기
    fi.loc[:n, :].plot.barh(y='importance_normalized', 
                            x='feature', color=color, 
                            edgecolor='k', figsize=figsize,
                            legend=False)

    plt.xlabel('Normalized Importance', size=18); plt.ylabel(''); 
    plt.title(f'Top {n} Most Important Features', size=18)
    plt.gca().invert_yaxis()

    new_output_path=f'{args.output_dir}{args.task_name}'
    write_path = os.path.join(new_output_path, "feature_importance.png")
    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path) 
    plt.savefig(write_path,bbox_inches='tight', pad_inches=0.5)
    plt.close()   
    
    return y_oof, test_preds, fi