import inference_for_serving as inference
import pandas as pd

## 테스트 코드 작성하는 곳

def main(args):
    user_data = pd.read_csv('/opt/ml/p4-dkt-ollehdkt/questions.csv')
    score = inference.inference(user_data,args)
    score = int(score)

if __name__ == "__main__":
    main(args)

def test_inference_for_serving(user_data,args):
    inference.inference(user_data,args)