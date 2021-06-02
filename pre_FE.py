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