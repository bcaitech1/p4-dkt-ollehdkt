from sqlalchemy import Column, Integer, String
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    password = db.Column(db.String(128))

    def __init__(self,id=None,password=None):
        self.id = id
        self.password = password

    def __repr__(self):
        return f'<User id : {self.id}>'