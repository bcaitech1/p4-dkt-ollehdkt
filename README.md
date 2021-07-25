# pstage_04_dkt(Deep Knowledge Tracing)
- 기간 : 2021.05.24~2021.06.15
- 대회 내용 : 학생의 지식 상태를 추적하여 문제 리스트 중 마지막 문제 정답 여부 예측(AUC : 0.8362 최종 7등/15팀 중) 
![task img](https://user-images.githubusercontent.com/52443401/126865028-66d9f100-e1c3-4633-8790-86c1f7d84f47.JPG)
### 수행 요약 
- config.yml파일을 통한 실험으로 베이스라인 코드의 편리함 증진
- lgbm을 기존 NN모델 파이프라인에 통합
- NN모델에 범주/연속형 피처를 자유롭게 넣을 수 있게 수정
- solve_time관련 feature들 생성
- user_split k-fold 구현
- 모든 모델 실험  

### Important Feature
- solve_time / correct_time, wrong_time
![fi 사진](https://user-images.githubusercontent.com/52443401/126864608-e6af562b-e2b0-4ad7-9c2f-7a86bbac5b98.png)


## config 파일을 통한 실행법
### 1. config.yml setting
모델 선택, 하이퍼 파라미터 선택, 기타 테크닉 옵션 선택

### 2. $ python3 train / inference .py
기존과 동일

### 3. $ python3 whole-in-one.py
학습-추론-제출 한번에 실행
단, lgbm은 inference를 따로 수행하지 않아도 됩니다. train부분에서 모두 처리
실행시 폴더에 학습 때 사용한 하이퍼 파라미터와 피처를 json으로 저장

### 4. $ python3 submit.py
key와 파일path를 입력하면 제출용 csv를 다운로드할 필요 없이 서버에 바로 제출
