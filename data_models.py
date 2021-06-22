from sqlalchemy import Column, Integer, String, Float
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# 사용자 테이블
class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    score = db.Column(db.Integer)

    def __init__(self,id=None,score=None):
        self.id = id
        self.score = score

    def __repr__(self):
        return f'<User id : {self.id}>'

# 로그 (테스트 용을 쓰일 예정임)
class ConnectLog(db.Model):
    __tablename__ = 'connectlog'

    id = db.Column(db.Integer,primary_key=True,autoincrement=True,unique=True) # log_id
    content = db.Column(db.String(128)) # 로그 내용
    logtime = db.Column(db.DateTime) # 로그 기록 시간

    def __init__(self,id,content=None,logtime=None):
        self.id=id
        self.content = content
        self.logtime = datetime.now()

    def __repr__(self): return '<Log %r>' % (self.content)

# 질문과 답
class Question(db.Model):
    __tablename__ = 'question'

    assessmentItemID = db.Column(db.String(20),primary_key=True) # 문항 번호
    testId = db.Column(db.String(20)) # 문제지 번호
    KnowledgeTag = db.Column(db.String(20)) # 문제 유형
    real_answer = db.Column(db.Integer) # 실제 답
    img_url = db.Column(db.String(256)) # 이미지 저장 경로
    q_content = db.Column(db.String(256)) # 문제 설명

    def __init__(self,assessmentItemID,testId=None,KnowledgeTag=None,real_answer=None,img_url=None,q_content=None):
        self.assessmentItemID = assessmentItemID
        self.testId = testId
        self.KnowledgeTag = KnowledgeTag
        self.real_answer = real_answer
        self.img_url = f'{img_url}'
        self.q_content = q_content if q_content is not None else ''
        # blob 방식으로 해도 되지만, sqlite는 경량 db이므로 url만 저장하고 이미지는 /static/problems 에 있다.


class Test(db.Model):
    __tablename__ = 'test'

    assessmentItemID = db.Column(db.String(20),primary_key=True) # 문항 번호
    testId = db.Column(db.String(20)) # 문제지 번호
    KnowledgeTag = db.Column(db.String(20)) # 문제 유형
    Timestamp = db.Column(db.DateTime)
    grade = db.Column(db.Integer)
    test_rate = db.Column(db.Float) 
    que_rate = db.Column(db.Float)
    tag_rate  = db.Column(db.Float)
    answerCode = db.Column(db.Integer)
    img_url = db.Column(db.Integer)
    q_content = db.Column(db.Integer)
    real_answer = db.Column(db.Integer)

    # 'Timestamp', 'KnowledgeTag', 'test_rate', 'que_rate',
    #    'tag_rate', 'answerCode'

    def __init__(self,assessmentItemID,testId=None,KnowledgeTag=None, Timestamp=None, grade = None, test_rate=None, que_rate=None, tag_rate=None, answerCode=None,
                img_url=None, q_content=None, real_answer=None):
        self.assessmentItemID = assessmentItemID
        self.testId = testId
        self.KnowledgeTag = KnowledgeTag
        self.Timestamp = Timestamp
        self.grade = grade
        self.test_rate = test_rate 
        self.que_rate = que_rate
        self.tag_rate  = tag_rate
        self.answerCode = answerCode
        self.img_url = img_url
        self.q_content = q_content
        self.real_answer = real_answer

    def to_dict(self):
        return {
           "assessmentItemID" : self.assessmentItemID, 
            "testId" : self.testId,
            "KnowledgeTag" : self.KnowledgeTag,
            "Timestamp" :        self.Timestamp, 
            "grade" :        self.grade,
            "test_rate" :       self.test_rate,
            "que_rate" :        self.que_rate,
            "tag_rate" :        self.tag_rate,
            "answerCode" : self.answerCode,
            "img_url" :        self.img_url,
            "q_content" : self.q_content,
            "real_answer" : self.real_answer
        }