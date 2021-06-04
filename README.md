# pstage_04_dkt
## config 파일을 통한 실행법
### 1. config setting
모델 선택, 하이퍼 파라미터 선택

### 2. $ python3 train / inference .py
기존과 동일

### 3. $ python3 whole-in-one.py
학습-추론 한번에 실행
단, lgbm은 inference를 따로 수행하지 않아도 됩니다. train부분에서 모두 처리
실행시 폴더에 학습 때 사용한 하이퍼 파라미터와 피처를 json으로 저장

### 4. $ python3 submit.py
key와 파일path를 입력하면 다운로드할 필요 없이 서버에서 바로 제출

## lgbm 합치기
### 1. __feature_engineering
