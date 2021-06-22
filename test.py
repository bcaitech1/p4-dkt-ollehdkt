import inference_for_serving as inference
import pandas as pd
from server import *

import random

import os
import sqlite3

## 테스트 코드 작성하는 곳

input_dir = f'/opt/ml/input/data/train_dataset'
db_dir = f'/opt/ml/code-final/p4-dkt-ollehdkt'

def main(args):
    # user_data = pd.read_csv('/opt/ml/p4-dkt-ollehdkt/questions.csv')
    # score = inference.inference(user_data,args)
    # score = int(score)

    # test_add_questions()
    test_list()
    # test_select_tag()
    # get_tags_test()

if __name__ == "__main__":
    main(args)

def test_inference_for_serving(user_data,args):
    inference.inference(user_data,args)

# 문제 추가 테스트
def test_add_questions():
    
    # QUESTION_LIST = [
    #                 {'q_content':'사람', 'tag':7597, 'test_id':'A010000060', 'assess_id':'A010060001'},
    #                 {'q_content':'동물', 'tag':397, 'test_id':'A050000094', 'assess_id':'A050094005'},
    #                 {'q_content':'자연', 'tag':451, 'test_id':'A050000155', 'assess_id':'A050155004'},
    #                 {'q_content':'물건', 'tag':587, 'test_id':'A060000017', 'assess_id':'A060017006'},
    #                 {'q_content':'실제', 'tag':4803, 'test_id':'A080000031', 'assess_id':'A080031001'},
    #             ]
    
    # new_question = Question(
    #     assessmentItemID = assessmentItemID,
    #     testId = testId,
    #     KnowledgeTag = KnowledgeTag,
    #     real_answer = real_answer,
    #     img_url = img_url,
    #     q_content = q_content
    # )

    QUESTION_LIST = [
                    {'tag':7597,'assess_id':'A010060001', 'test_id':'A010000060', "real_answer":2,"img_url":'problem15.jpg',"q_content":'2021 수능 가형 15번'},
                    {'tag':397, 'test_id':'A050000094', 'assess_id':'A050094005',"real_answer":3,"img_url":'problem17.jpg',"q_content":'2021 수능 가형 17번'},
                    {'tag':451, 'test_id':'A050000155', 'assess_id':'A050155004',"real_answer":3,"img_url":'problem18.jpg',"q_content":'2021 수능 가형 18번'},
                    {'tag':587, 'test_id':'A060000017', 'assess_id':'A060017006',"real_answer":5,"img_url":'problem20.jpg',"q_content":'2021 수능 가형 20번'},
                    {'tag':4803, 'test_id':'A080000031', 'assess_id':'A080031001',"real_answer":2,"img_url":'problem21.jpg',"q_content":'2021 수능 가형 21번'},
                ]

    for q in QUESTION_LIST:
        add_question(
            assessmentItemID = q['assess_id'],
            testId = q['test_id'],
            KnowledgeTag = q['tag'],
            real_answer = q['real_answer'],
            img_url = q['img_url'],
            q_content = q['q_content']
        )
    print('Done...')

# csv로 sqlite에 저장하는 테스트
def test_list():
    csv_file = os.path.join(input_dir,'test_list.xlsx')
    df = pd.read_excel(csv_file)
    df.set_index('assessmentItemID',inplace = True)
    # df['img_url'] = 'no-image.jpg'
    df['img_url'] = df['img_url'].apply(lambda x:f'{x[:-4]}.png')
    print(type(df['img_url']))
    # df['q_content'] = [f'문제 : {i}' for i in range(len(df))]
    print(df)
    print(df.columns)
    df['real_answer']=1
    df['real_answer'] = df['real_answer'].apply(lambda x: random.randint(1,5))

    con = sqlite3.connect(os.path.join(db_dir,'db.sqlite'))
    df.to_sql('test',con)

    pass

# 2065, 2085, 7600, 7621, 2010
def test_select_tag():
    get_question_by_tag(2065)
    pass

def get_tags_test():
    res = get_tags()
    print(res)
    pass