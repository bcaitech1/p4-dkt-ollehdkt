def user_tag_ansrate_feature(df):
    tag_ansrate=[]
    tag_len=[]
    for uid in df['userID'].unique():
        interactions=df[df['userID']==uid]
        user_tag_dict=defaultdict(list)
        for idx in range(len(interactions)):
            tag=interactions.iloc[idx]['KnowledgeTag']
            answer=interactions.iloc[idx]['answerCode']
            if idx==0 or len(user_tag_dict[tag])==0 :
                tag_ansrate.append(0)
            else :
                tag_ansrate.append(sum(user_tag_dict[tag])/len(user_tag_dict[tag]))
            

            tag_len.append(len(user_tag_dict[tag]))    
            user_tag_dict[tag].append(0 if answer==-1 else answer)
            
    print(len(tag_ansrate))
    print(len(tag_len))
    #유저가 현재 풀고있는 문제 유형을 몇번이나 풀었고 그에 따른 정답률를 리턴한다 
    return tag_ansrate,tag_len

def total_tag_ans_rate_feature(df):
    #태그별 정답률
    tag_groupby = df.groupby('KnowledgeTag').agg({
        'answerCode': percentile
    }).reset_index(drop=False)
    tag_groupby
    tag_ansrate=zip(tag_groupby['KnowledgeTag'],tag_groupby['answerCode'])
    tag_ansrate_dict=dict(list(tag_ansrate))
    return df['KnowledgeTag'].apply(lambda x:tag_ansrate_dict[x])

#사용자의 문제풀이 시간을 구하는 함수
def make_solve_time(df):
    def convert_time(s):
        timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
        return int(timestamp)
    df=df.copy()
    df['sec_time']=df['Timestamp'].apply(convert_time)
    df['solve_time']=df['sec_time']-df['sec_time'].shift(1)
    return df


#agg에서 첫번째 값을 보간하기 위한 함수
def get_interpolate(s):
    s=s.values
#     print(s[0])
    if s[0]>3600 or pd.isnull(s[0]) or s[0]<0:
        s[0]=np.nan
        s=pd.Series(s)
        s=s[::-1].interpolate()[::-1]
    
    return s[0]

#df에 solve_time을 보간하여 추가하는 함수
def make_timecv(df):
    answer=pd.DataFrame()
    for user in df['userID'].unique():
        interactions=df[df['userID']==user]
        interactions.sort_values(by=['testId','Timestamp'], inplace=True)
        inter_df=interactions.groupby('testId').agg({'solve_time':lambda x: get_interpolate(x)}).reset_index(drop=False) 
#         print(inter_df)
        #보간한 테스트 아이디별 시간 dict로 저장
        inter_time_dict=dict(zip(inter_df['testId'],inter_df['solve_time']))
        need_inter_interactions=interactions[interactions['testId']!=interactions['testId'].shift(1)]
        need_inter_interactions['solve_time']=need_inter_interactions['testId'].apply(lambda x:inter_time_dict[x])
        under_interactions=interactions[interactions['testId']==interactions['testId'].shift(1)]
        total_user=pd.concat([need_inter_interactions,under_interactions], ignore_index=False)
        answer=pd.concat([answer,total_user], ignore_index=False)
    answer.sort_values(by=['userID','Timestamp'], inplace=True)
    return answer

#문제별 풀이시간 평균을 구하는 함수
def get_time_avg_dict():
    csv_file_path = '/opt/ml/input/data/train_dataset/train_time_fixed.csv'
    df = pd.read_csv(csv_file_path) 
    csv_file_path = '/opt/ml/input/data/train_dataset/test_time_fixed.csv'
    tdf = pd.read_csv(csv_file_path) 
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    # df.sort_values(by=['userID','Timestamp'], inplace=True)
    total_df=pd.concat([df,tdf],ignore_index=True)
    time_mean_df=total_df.groupby('assessmentItemID').agg({'solve_time':'mean'}).reset_index(drop=False)
    item_time_dict=dict(zip(time_mean_df['assessmentItemID'],time_mean_df['solve_time']))
    return item_time_dict

def make_wrongtime_right(df,item_time_dict):
    def maket(t):
        time=t['solve_time']
        if time>3600 or time<0:
            t['solve_time']=item_time_dict[t['assessmentItemID']]
        return t['solve_time']
    df['solve_time']=df.apply(maket,axis=1)
    return df