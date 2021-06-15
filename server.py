from flask import Flask, render_template
from flask import request

from data_models import db

import inference_for_serving as inference

 
app = Flask(__name__)
args = None
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/menu',methods=['GET'])
def menu():
    return render_template('menu.html')

@app.route('/get_score', methods=['POST'])
def get_score():
    data = request.json
    user_data = []
    print(data)
    for d in data:
        if 'answer' in d:
            row = [d['assess_id'], d['test_id'],d['tag'], d['answer']]
            user_data.append(row)
     
    print(user_data)
    score = inference.inference(user_data,args)
    score = int(score)
    return str(score)

def main(args_input):
    global args
    args = args_input
    app.run(host="0.0.0.0", port=6006, debug=True)
    return args_input

# if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=6006, debug=True)