## Boost Camp - AI Tech

> ### Stage 4 - Deep Knowledge Tracing

> 2021.05.24 ~ 2021.06.15
>
> 특정 문제를 푼 사용자의 마지막 정답 여부 예측 문제 

`Boost Camp P stage 4 대회의 과정과 결과를 담은 Git repo 입니다. 대회 규칙상 특정 내용이 수정되거나 삭제된 경우가 존재합니다`

---

<br>

### Final Score 🏁

Team Rank : 7 , AUROC : 0.8362, Accuracy : 0.7527

<br>

### Table of content 📋

##### [Olleh Team 소개](#team)<br>

##### [프로젝트 전체 과정](#process)<br>

##### [핵심 전략](#strategy)<br>

##### [성능 향상을 위한 고군분투한 여정 🏃‍♀️](#fullprocess)

1. [EDA](#EDA)<br>
2. [Feature Engineering](#FE)<br>
3. [Data Augmentation](#aug)<br>
4. [Model](#model)<br>
5. [Cross Validation strategy](#CV)<br>
6. [기타](#etc)<br>

<br>

---

---

### Olleh Team <a name = 'team'></a>



#### 김종호 ![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/Headbreakz)![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://philgineer.com/)

#### 박상기 ![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/sangki930)![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://sangki930.tistory.com/)

#### 스후페엘레나

#### 임도훈 ![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://blog.naver.com/vail131)![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://philgineer.com/)

#### 지정재 ![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/PrimeOfMIne)![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://www.notion.so/bf6a15f41ccf4d5b9e5d056915cf2793)

#### 홍채원 ![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://www.notion.so/P_stage-4-6cfb7db7ddc8400b9e58a7eb1f70d13f)

#### 

#### 

<br>

---

### 프로젝트 전체 과정 📖<a name = 'process'></a>

![image1](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/headbreakz/image/total_process.png?raw=true)

<br>

---

<br>

### 핵심 전략💡<a name = 'strategy'></a>

 교육 도메인 지식 활용

 user split augmentation

 private leader board를 고려한 모델 실험

 two track (task cross-reference)

<br>

---

<br>

### 성능 향상을 위한 고군분투한 여정 🏃‍♀️<a name ='fullprocess'></a>

#### 1. EDA (Exploratory Data Analysis)<a name='EDA'></a>

 다양한 EDA를 통해 Feature engineering과 validation 전략을 세우는데 활용

![image](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/headbreakz/image/EDA.gif?raw=true)

<br>

### 2.Feature Engineering<a name ='FE'></a>

 데이터 분석 기반 Feature

 User ID, assessmentItemID, testId, KnowledgeTag, Timestamp 과 answerCode 관계

 각 Value와 answerCode값의 평균, 분산, Skew, 누적합, 누적 평균

 각 Value 값의 통계적 수치

 교육학 이론 기반 Feature

assessmentItemID, testId, KnowledgeTag의 변별도 값 

변별도 : (상위 정답 수 - 하위 정답 수 ) / (총 응시자 / 2)

ELO rating

정답 여부에 따른 개인 Rank 점수 적용

문제 난이도에 따른 Rank 점수의 증가와 감소

총 47개의 Feature 생성

[Feature Engineering 상세](https://www.notion.so/Feature-Engineering-0189914b580a483083b88982006984d6)

![image](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/headbreakz/image/FE.png?raw=true)

<br>

#### 3. Data augmentation <a name = 'aug'></a>

Sliding Window(Stride = 10,20, ... ,128)

User month split (사용자를 월별로 정리)

User testID grade split (사용자를 문제지별 정리)

<br>

#### 4. Model <a name = 'model'></a>



➡ Tree decision : LGBM , XGBoost , Catboost

➡ NN Models : LSTM , LSTM with Attention , Bert , Saint , GPT-2, LastQuery_pre/post

![image4](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/headbreakz/image/model.png?raw=true)

<br>

#### 5.Cross validation strategy <a name = 'CV'></a>

이전 stage에서 shake-up이 크게 일어나서 큰 점수 하락을 겪었기 때문에 validation 전략에 조금 더 신경을 썼습니다.

➡ UserID split

✳ userID를 기준으로 k-fold를 진행

LGBM은 NN과 다르게 interaction을 기준으로 학습 진행

⇒ LGBM도 NN 처럼 userID 를 기준으로 k-fold validation을 진행

➡ grade별 검증

✳ 사용자의 대표 grade를 추출하여, grade의 비율에 맞게 K-fold 수행

✳ A**030**071005

testID, AssesmentID 에서 앞자리 3자리의 경우 Grade

**BUT!** user 별로 grade가 고정되어 있지 않다.

(ex. userID 315가 grade 3, 4, 7의 문제를 모두 푼다.)

따라서 사용자의 grade를 하나로 특정하기 어려운 문제 발생

⇒ 하나의 사용자에서 가장 많이 등장한 grade를 기준으로 사용자의 대표 grade 설정

설정한 대표 grade를 기준으로 train set 과 test set의 분포가 유사

⇒ grade의 분포 비율을 유지시켜서, train의 분포와 test의 분포가 유사하도록  validation을 진행

 <br>

#### 6. 기타 <a name = 'etc'></a>

➡ Hyper parameter tuning - Optuna

➡ ensemble

✳ soft voting

분류기들의 레이블 값 결정 확률을 모두 더하고 **이를 평균**해서 이들 중 확률이 가장 높은 레이블 값을 최종 보팅 결괏값 으로 선정.



일반적으로는 soft voting을 적용하는 경우가 多

DKT competetion의 eval-metric이 AUC 이므로 class label값을 제출하는 것이 아닌 확률값을 제출하므로, 모델별 prediction 값을 평균내는 부분까지만 진행

→ 결과 : 단일 모델 결과보다 PB LB 점수상으로 하락

하락 근거 : 모델마다 prediction 값이 다른 것으로 판단 (NN, tree 모델을 soft voting하는 경우 이런 경우가 잦다고 알려져 있음) ⇒ oof-stacking을 시도해야할 것이라고 판단

✳ hard voting

다수결, 여러 모델의 결과 값을 기준으로 가장 많은 모델이 예측한 class label을 기준으로 예측값을 도출

DKT competetion의 eval-metric이 AUC 이므로 class label값을 제출하는 것이 아닌 확률값을 제출하므로, 가장 많은 모델이 예측한 class label로 예측한 모델들의 prediction 값을 평균낸 값으로 제출

→ 결과 : 단일 모델 결과보다 PB LB 점수상으로 하락

✳ oof_stacking

다양한 ensemble 방법 중 oof-stacking 실험

stacking : 다양한 model들의 예측결과를 결합해 최종 예측 결과를 만들어 내는 것

근거 : NN기반 model(LSTM, transformer등 sequential한 data 처리를 위한 NN model)의 prediction 결과값과 tree 기반의 model의 prediction 결과 값이 상이한 것을 위의 soft voting의 결과로써 얻었기 때문에 이렇게 결과가 상이한 경우 메타 모델을 통해 ensemble을 하게되는 oof-stacking 방법이 효과적으로 알고 있었기에 이를 진행하고자 하였다.

✳ Priority Max Ensemble

- 상위 4개 prediction 중 정확도가 가장 높은 정확도를 가진 prediction을 우선으로하고 prediction에 max값을 취해서 Ensemble
- 정확도 값은 보존하면서 auc가 높아질 가능성이 높아 선택

### 