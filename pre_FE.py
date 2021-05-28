def user_tag_ansrate_feature(df):
    main_arr=[]
    for uid in df['userID'].unique():
        interactions=df[df['userID']==uid]
        user_tag_dict=defaultdict(list)
        for idx in range(len(interactions)):
            tag=interactions.iloc[idx]['KnowledgeTag']
            answer=interactions.iloc[idx]['answerCode']
            if idx==0 or len(user_tag_dict[tag])==0 :
                main_arr.append(0)
            else :
                main_arr.append(sum(user_tag_dict[tag])/len(user_tag_dict[tag]))

            user_tag_dict[tag].append(answer)
    print(len(main_arr))
    return main_arr

def total_tag_ans_rate_feature(df):
    #태그별 정답률
    tag_groupby = df.groupby('KnowledgeTag').agg({
        'answerCode': percentile
    }).reset_index(drop=False)
    tag_groupby
    tag_ansrate=zip(tag_groupby['KnowledgeTag'],tag_groupby['answerCode'])
    tag_ansrate_dict=dict(list(tag_ansrate))
    return df['KnowledgeTag'].apply(lambda x:tag_ansrate_dict[x])