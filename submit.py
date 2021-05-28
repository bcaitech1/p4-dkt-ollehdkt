import json
import os
import requests

def submit(user_key='', file_path = ''):
    if not user_key:
        raise Exception("No UserKey" )
    url = 'http://ec2-13-124-161-225.ap-northeast-2.compute.amazonaws.com:8000/api/v1/competition/42/presigned_url/?description=&hyperparameters={%22training%22:{},%22inference%22:{}}'
    headers = {
        'Authorization': user_key
    }
    res = requests.get(url, headers=headers)
    print(res.text)
    data = json.loads(res.text)
    
    submit_url = data['url']
    body = {
        'key':'app/Competitions/000042/Users/{}/Submissions/{}/output.csv'.format(str(data['submission']['user']).zfill(8),str(data['submission']['local_id']).zfill(4)),
        'x-amz-algorithm':data['fields']['x-amz-algorithm'],
        'x-amz-credential':data['fields']['x-amz-credential'],
        'x-amz-date':data['fields']['x-amz-date'],
        'policy':data['fields']['policy'],
        'x-amz-signature':data['fields']['x-amz-signature']
    }
    requests.post(url=submit_url, data=body, files={'file': open(file_path, 'rb')})

if __name__ == "__main__":
    test_dir='/opt/ml'#prediction folder path
    
    # 아래 글을 통해 자신의 key값 찾아 넣기
    # http://boostcamp.stages.ai/competitions/3/discussion/post/110
    submit("Bearer 15bdf505e0902975b2e6f578148d22136b2f7717", os.path.join(test_dir, 'answer.csv'))