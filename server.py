from flask import Flask, render_template
from flask import request,jsonify

from data_models import *

import inference_for_serving as inference

import os

from datetime import datetime

import pandas as pd

import json


app = Flask(__name__,static_folder='/opt/ml/code-final/p4-dkt-ollehdkt/static',instance_relative_config=True)
args = None

basdir = os.path.abspath(os.path.dirname(__file__))
dbfile = os.path.join(basdir, 'db.sqlite')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + dbfile
print(f'db_path : {dbfile} // {basdir}')
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # 추가적인 메모리를 필요로 하므로 False
app.config['SECRET_KEY'] = 'jqiowejrojzxcovnklqnweiorjqwoijroi'

db.init_app(app)
db.app = app
db.create_all()

@app.route('/', methods=['GET'])
def index():
    print(f'성공여부 : {add_log()}') # 성공여부
    return render_template('index.html',questions=get_question())

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
    
    return str(score)

# @app.route('/get_question',methods=['GET'])
def get_question():
    data = db.session.query(Question).all() # 모든 레코드 조회
    
    # df = pd.read_sql(data.statement, data.session.bind)
    # 옵션으로 orient 를 준다
    
    # print(json.loads(df.to_json(orient='records')))
    # return jsonify(json.loads(df.to_json(orient='records')))
    print(type(data))
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