# Boost Camp - AI Tech

> ## Stage 4 - Deep Knowledge Tracing

> ### 2021.05.24 ~ 2021.06.15
>
> ### 특정 문제를 푼 사용자의 마지막 정답 여부 예측 문제 

`Boost Camp P stage 4 대회의 과정과 결과를 담은 Git repo 입니다. 대회 규칙상 특정 내용이 수정되거나 삭제된 경우가 존재합니다`

---

<br>

### 🏁 Final Score 

Team Rank : 7 , AUROC : 0.8362, Accuracy : 0.7527

<br>

---



### 📋 Table of content 

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

### 🌏Olleh Team <a name = 'team'></a>

#### 김종호 [Project Branch](https://github.com/bcaitech1/p4-dkt-ollehdkt/tree/headbreakz) ![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/Headbreakz)![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://philgineer.com/) 

#### 박상기 [Project Branch](https://github.com/bcaitech1/p4-dkt-ollehdkt/tree/final_sangi) ![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/sangki930)![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://sangki930.tistory.com/)

#### 임도훈 [Project Branch](https://github.com/bcaitech1/p4-dkt-ollehdkt/tree/final_dh) ![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://blog.naver.com/vail131)![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://philgineer.com/) 

#### 지정재 [Project Branch](https://github.com/bcaitech1/p4-dkt-ollehdkt/tree/comb_main) ![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/PrimeOfMIne)![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://www.notion.so/bf6a15f41ccf4d5b9e5d056915cf2793) 

#### 홍채원 [Project Branch](https://github.com/bcaitech1/p4-dkt-ollehdkt/tree/headbreakz) ![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/hcw3737)![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://www.notion.so/P_stage-4-6cfb7db7ddc8400b9e58a7eb1f70d13f)

#### 스후페엘레나

`Project Branch는 DKT 대회에서 사용한 팀원 별 Branch입니다. 팀원의 자세한 정보를 원하시는 경우 Project Branch로 확인 가능합니다` 

<br>

---

<br>

### 📖프로젝트 전체 과정 <a name = 'process'></a>

![image1](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/headbreakz/image/total_process.png?raw=true)

<br>

---

<br>

### 💡 핵심 전략<a name = 'strategy'></a>

 ➡교육 도메인 지식 활용

 ➡User split augmentation

 ➡private leader board를 고려한 모델 실험

 ➡Two track (task cross-reference)

<br>

---

<br>

### 🏃‍♀️ 성능 향상을 위한 고군분투한 여정 <a name ='fullprocess'></a>

### 1. EDA (Exploratory Data Analysis)<a name='EDA'></a>

➡ 다양한 EDA를 통해 Feature engineering과 validation 전략을 세우는데 활용

![image](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/headbreakz/image/EDA.gif?raw=true)

<br>

### 2.Feature Engineering<a name ='FE'></a>

 ➡ 데이터 분석 기반 Feature

　✳ User ID, assessmentItemID, testId, KnowledgeTag, Timestamp 과 answerCode 관계

　✳각 Value와 answerCode값의 평균, 분산, Skew, 누적합, 누적 평균

　✳각 Value 값의 통계적 수치

 ➡ 교육학 이론 기반 Feature

　✳assessmentItemID, testId, KnowledgeTag의 변별도 값 

　✳변별도 : (상위 정답 수 - 하위 정답 수 ) / (총 응시자 / 2)

➡ ELO rating

　✳정답 여부에 따른 개인 Rank 점수 적용

　✳문제 난이도에 따른 Rank 점수의 증가와 감소

➡ 총 47개의 Feature 생성

➡ [Feature Engineering 상세](https://www.notion.so/Feature-Engineering-0189914b580a483083b88982006984d6)

![image](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/headbreakz/image/FE.png?raw=true)

<br>

### 3. Data augmentation <a name = 'aug'></a>

➡ Sliding Window(Stride = 10,20, ... ,128)

➡ User month split (사용자를 월별로 정리)

➡ User testID grade split (사용자를 문제지별 정리)

<br>

### 4. Model <a name = 'model'></a>

➡ Tree decision : LGBM , XGBoost , Catboost

➡ NN Models : LSTM , LSTM with Attention , Bert , Saint , GPT-2, LastQuery_pre/post

![image4](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/headbreakz/image/model.png?raw=true)

<br>

### 5.Cross validation strategy <a name = 'CV'></a>

`이전 stage에서 shake-up이 크게 일어나서 큰 점수 하락을 겪었기 때문에 validation 전략에 조금 더 신경을 썼습니다.`

➡ UserID split

　✳userID를 기준으로 k-fold를 진행

​	

➡ grade별 검증

　✳ 사용자의 대표 grade를 추출하여, grade의 비율에 맞게 K-fold 수행

　✳ ex) A**030**071005, testID, AssesmentID 에서 앞자리 3자리의 경우 Grade

　✳ 상세 

  User 별로 grade가 고정되어 있지 않은 경우를 확인하였다.( ex, userID 315가 grade 3, 4, 7의 문제를 모두 푸는 경우) 따라서 사용자의 grade를 하나로 특정하기 어려운 문제 발생하였다. 이를 해결 하기 위해 하나의 사용자에서 가장 많이 등장한 grade를 기준으로 사용자의 대표 grade 설정하였다.

  설정한 대표 grade를 기준으로 데이터 분포를 확인 한 결과 Train set 과 Test set의 분포가 유사하다는 것을 확인하였다.

![image5](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/headbreakz/image/data.png?raw=true)

 <br>

### 6. 기타 <a name = 'etc'></a>

➡ Hyper parameter tuning - Optuna

➡ PCA

　✳ 40개의 features를 input으로 하여 주성분 분석을 수행	

　✳ 주성분 분석 결과 - 20개의 주성분

![image10](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/chaewon/image/PCA.png?raw=true)

➡ ensemble

　✳ soft voting

  분류기들의 레이블 값 결정 확률을 모두 더하고 **이를 평균**해서 이들 중 확률이 가장 높은 레이블 값을 최종 보팅 결괏값 으로 선정.

![image8](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/headbreakz/image/etc.png?raw=true)



  DKT competetion의 eval-metric이 AUC 이므로 class label값을 제출하는 것이 아닌 확률값을 제출하므로, 모델별 prediction 값을 평균내는 부분까지 진행하였고, 단일 모델 결과보다 PB LB 점수상으로 하락하였다.

  모델마다 상이한 prediction으로 인해 값이 하락한 것으로 판단되어 OOF stacking을 시도해야 할것으로 판단하였다. 

<br>

✳ hard voting

  다수결, 여러 모델의 결과 값을 기준으로 가장 많은 모델이 예측한 class label을 기준으로 예측값을 도출하였다. 

  DKT competetion의 eval-metric이 AUC 이므로 class label값을 제출하는 것이 아닌 확률값을 제출하므로, 가장 많은 모델이 예측한 class label로 예측한 모델들의 prediction 값을 평균낸 값으로 제출하였고, soft voting과 동일하게 단일 모델 결과보다 PB LB 점수상으로 하락을 보였다.

<br>

✳ oof_stacking

 NN기반 model의 prediction 결과값과 tree 기반의 model의 prediction 결과 값이 상이한 것을 위의 soft voting의 결과로써 얻었기 때문에 이렇게 결과가 상이한 경우 메타 모델을 통해 ensemble을 하게되는 oof-stacking 방법이 효과적으로 알고 있었기에 이를 진행하고자 하였다.

![image8](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/headbreakz/image/OOF.png?raw=true)

![image8](https://github.com/bcaitech1/p4-dkt-ollehdkt/blob/headbreakz/image/oof2.png?raw=true)

<br>

✳ Priority Max Ensemble

상위 4개 prediction 중 정확도가 가장 높은 정확도를 가진 prediction을 우선으로 4개의 prediction의 max값을 취해서 Ensemble을 하였다. 이러한 방법을 선택한 이유는 정확도 값은 보존하면서 auc가 높아질 것으로 예상하여 사용하였다.

```python
pd_list[0]['prediction']
new_df = pd.DataFrame(columns=['prediction'])
for i in range(len(pd_list[0])):
    id=i
    a1 = pd_list[0]['prediction'][i] 
    a2 = pd_list[1]['prediction'][i] # 가장 높은 acc를 가진 prediction(이하 1번예측)
    a3 = pd_list[2]['prediction'][i]
    a4 = pd_list[3]['prediction'][i]

    d = {"up":[],"down":[]}

    for j in range(4): # 0.5를 기준으로 나눈다.
        if pd_list[j]['prediction'][i]>=0.5:
            d["up"].append(j)
        else:
            d["down"].append(j)

    if len(d["up"])>0 and len(d["down"])>0: 
        # 0.5를 기준으로 up, down이 있을 때, prediction을 max로 하여 auc를 늘림
        # 1번 예측이 어느 그룹에 포함되어 있을 때, 그 그룹에서 max 취함
        if (1 in d["up"]):
            m = pd_list[max(d["up"])]['prediction'][i]
        elif (1 in d["down"]):
            m = pd_list[max(d["down"])]['prediction'][i]
    else: # 네 개다 up 또는 down에 모두 있으면, max로 prediction 값을 구함
        m=(max(pd_list[0]['prediction'][i],pd_list[1]['prediction'][i],pd_list[2]['prediction'][i],pd_list[3]['prediction'][i]))
    
    new_df.loc[len(new_df)]=[m]
```

<br>

---

### Reference

[Deep Knowledge Tracing](https://arxiv.org/pdf/1506.05908.pdf)

[BERT](https://arxiv.org/abs/1810.04805)

[Bayesian Opitimization](https://arxiv.org/pdf/1807.02811.pdf)

[Saint+](https://arxiv.org/pdf/2010.12042.pdf)

[EGNET+KT1](https://arxiv.org/pdf/1912.03072.pdf)

---

