from flask import Flask, render_template
from flask import request,jsonify

from data_models import *

import inference_for_serving as inference

import os

from datetime import datetime

import pandas as pd

import json

import random


app = Flask(__name__,static_folder='/opt/ml/code-final/p4-dkt-ollehdkt/static',instance_relative_config=True)
args = None

basdir = os.path.abspath(os.path.dirname(__file__))
dbfile = os.path.join(basdir, 'db.sqlite')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + dbfile
print(f'db_path : {dbfile} // {basdir}')
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # 추가적인 메모리를 필요로 하므로 False
app.config['SECRET_KEY'] = 'jqiowejrojzxcovnklqnweiorjqwoijroi'
app.config['JSON_AS_ASCII'] = False

db.init_app(app)
db.app = app
db.create_all()

@app.route('/', methods=['GET'])
def index():
    print(f'로그 삽입 성공여부 : {add_log()}') # 성공여부
    return render_template('index.html',questions=get_questions())

@app.route('/menu',methods=['GET'])
def menu():
    return render_template('menu.html')

@app.route('/join',methods=['GET'])
def join():
    return render_template('join.html')

@app.route('/graph_sample',methods=['GET'])
def graph_sample():
    return render_template('graph-sample/index.html')

# static 폴더 지정
@app.route('/')
def test_page(path):
    return send_from_directory(app.static_folder,request.path[1:])

@app.route('/get_score', methods=['POST'])
def get_score():
    data = request.json
    user_data = []
    print(data)
    for d in data:
        if 'answer' in d:
            row = [d['assessmentItemID'], d['testId'],d['KnowledgeTag'], d['answer']]
            user_data.append(row)
     
    print(user_data)
    score = inference.inference(user_data,args)
    score = int(score)

    second_result = inference.inference(user_data[5:],args)
    second_result = int(second_result)
    
    return jsonify({
        "score" : score,
        "second_result" : second_result
    })

# @app.route('/get_question',methods=['GET'])
# def get_question():
#     res=[]
#     for i in get_questions():
        
#         d = {
#             "assessmentItemID" : i.assessmentItemID,
#             "testID" : i.testId,
#             "KnowledgeTag" : i.KnowledgeTag,
#             "real_answer" : i.real_answer,
#             "img_url" : i.img_url,
#             "q_content" : i.q_content
#         }
#         res.append(d)
        
#     return jsonify(res)

@app.route('/get_question',methods=['GET'])
def get_question():
    tag = int(request.args.get('tag',0))
    start = int(request.args.get('start',0))
    end = int(request.args.get('end',0))
    res=[]
    # print(get_random_question(start,start+5,tag))
    for i in get_random_question(start,end,tag):
        res.append(i.to_dict())
        
    return jsonify(res)

@app.route('/get_tags',methods=['GET'])
def get_tags_for_web():
    d = {"data" : get_tags()}
    return jsonify(d)

# @app.route('/get_question/<tag>',methods=["GET"])
# def get_question_with_tag(tag):


#     return

# @app.route('/get_question',methods=['GET'])
def get_questions():
    data = db.session.query(Question).all() # 모든 레코드 조회
    
    # df = pd.read_sql(data.statement, data.session.bind)
    # 옵션으로 orient 를 준다
    
    # print(json.loads(df.to_json(orient='records')))
    # return jsonify(json.loads(df.to_json(orient='records')))
    print(f'데이터 타입 : {type(data)}')
    return data


def main(args_input):
    global args
    args = args_input
    app.run(host="0.0.0.0", port=6006, debug=True)
    return args_input

# if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=6006, debug=True)

## log 추가
def add_log():
    new_log = ConnectLog(
        id=None,
        content='접속 확인',
        logtime = datetime.now()
    )
    try:
        db.session.add(new_log)
        db.session.commit()
    except:
        return 0 # 실패

    return 1


# 문제 추가
def add_question(
    assessmentItemID = None, # 문항 번호
    testId = None,
    KnowledgeTag = None,
    real_answer = None,
    img_url = None,
    q_content = None
):

    new_question = Question(
        assessmentItemID = assessmentItemID,
        testId = testId,
        KnowledgeTag = KnowledgeTag,
        real_answer = real_answer,
        img_url = img_url,
        q_content = q_content
    )
    try:
        db.session.add(new_question)
        db.session.commit()
    except:
        return 0 # 실패

    return 1 # 성공

def get_question_by_tag(tag : int):
    try:
        q=db.session.query(Test).filter(Test.KnowledgeTag==tag).all()
        print(type(q))
        print(q)
    except:
        print("예외 발생")
        return None
    
    return q

def get_random_question(start:int,end:int,tag:int):

    res = get_question_by_tag(tag)
    random.shuffle(res)

    return res[start:end]

def get_tags():
    try:
        q=db.session.query(Test.KnowledgeTag).distinct().all()      
    except:
        print("예외 발생")
        return None
    q = [i[0] for i in q]
    return q
