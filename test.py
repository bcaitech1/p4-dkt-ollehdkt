import inference_for_serving as inference
import pandas as pd
from server import *

## 테스트 코드 작성하는 곳

def main(args):
    # user_data = pd.read_csv('/opt/ml/p4-dkt-ollehdkt/questions.csv')
    # score = inference.inference(user_data,args)
    # score = int(score)

    test_add_questions()

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

def test01():
    pass