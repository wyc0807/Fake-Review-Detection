#import pymysql
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from collections import defaultdict
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split#,GridSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC  
from sklearn import cross_validation,metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, classification_report, roc_curve, auc, accuracy_score  

df=pd.read_csv('info_addFeatures_new_v2.csv',encoding='gb18030')

train, test = train_test_split(df, test_size=0.2)
x_train = [[] for _ in range(len(train))]
y_train = []
   
x_test = [[] for _ in range(len(test))]
y_test = []

print('start')
#user_review_count	user_useful	user_funny	user_cool	user_fans	user_average_stars	user_sum_compliment	uc_review_count	uc_avg_stars
#uc_avg_review_useful	uc_avg_review_funny	uc_avg_review_coo
ini = 0
train, test = train_test_split(df, test_size=0.2)
x_train = [[] for _ in range(len(train))]
y_train = []
    
x_test = [[] for _ in range(len(test))]
y_test = []
ini = 0
for index, row in train.iterrows():
    #print(ini, len(train), ini/len(train))
    x_train[ini].extend((row['rating'],row['usefulCount_review'],row['friendCount'],
                               row['reviewCount'],row['usefulCount'],row['coolCount'],
                               row['funnyCount'],row['complimentCount'],row['fanCount'],
                               row['tipCount'],row['AverageReviewLengthOfReviewer'],row['MaxReviewSimilarityOfReviewer']
                               ,row['ExetremeRatingRatioOfReviewer'],row['MaxReviewNumsPerDayOfReviewer']
                               ,row['RatingDeviationFromMeanRating'],row['TimeRatioOfReviewInAllReviews']))
    if row['flagged'] == 'Y':
        y_train.append(1)
    else:
        y_train.append(0)
    ini += 1 
print('training features completed') 
ini = 0
for index, row in test.iterrows():
    #print(ini, len(test), ini/len(test))
    x_test[ini].extend((row['rating'],row['usefulCount_review'],row['friendCount'],
                               row['reviewCount'],row['usefulCount'],row['coolCount'],
                               row['funnyCount'],row['complimentCount'],row['fanCount'],
                               row['tipCount'],row['AverageReviewLengthOfReviewer'],row['MaxReviewSimilarityOfReviewer']
                               ,row['ExetremeRatingRatioOfReviewer'],row['MaxReviewNumsPerDayOfReviewer']
                               ,row['RatingDeviationFromMeanRating'],row['TimeRatioOfReviewInAllReviews']))
    if row['flagged'] == 'Y':
        y_test.append(1)
    else:
        y_test.append(0)
    
    ini += 1 
print('testing features completed')

x_train = pd.DataFrame(x_train, columns = list(['rating','usefulCount_review','friendCount',
                               'reviewCount','usefulCount','coolCount',
                               'funnyCount','complimentCount','fanCount',
                               'tipCount','AverageReviewLengthOfReviewer','MaxReviewSimilarityOfReviewer'
                               ,'ExetremeRatingRatioOfReviewer','MaxReviewNumsPerDayOfReviewer'
                               ,'RatingDeviationFromMeanRating','TimeRatioOfReviewInAllReviews']
                    ))
y_train = pd.DataFrame(y_train, columns = list("a"))
x_test = pd.DataFrame(x_test, columns = list(['rating','usefulCount_review','friendCount',
                               'reviewCount','usefulCount','coolCount',
                               'funnyCount','complimentCount','fanCount',
                               'tipCount','AverageReviewLengthOfReviewer','MaxReviewSimilarityOfReviewer'
                               ,'ExetremeRatingRatioOfReviewer','MaxReviewNumsPerDayOfReviewer'
                               ,'RatingDeviationFromMeanRating','TimeRatioOfReviewInAllReviews']
                    ))
y_test = pd.DataFrame(y_test, columns = list("a"))

### scale the data first
scaler = preprocessing.StandardScaler().fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
#Xgb models-----------------------------------------------------------------------------------------    
###xgboost
print('Start Xgb')
x_train_Xgb = x_train
y_train_Xgb = y_train['a']
x_test_Xgb = x_test
y_test_Xgb = y_test['a']
#print (y_test_Xgb)
   


xgb_model = xgb.XGBClassifier(learning_rate = 0.3, max_depth = 6, min_child_weight = 1, gamma = 0.1).fit(x_train_Xgb, y_train_Xgb)
pre_Xgb = xgb_model.predict(x_test_Xgb)
print(pre_Xgb)

t=accuracy_score(y_test,pre_Xgb)
pred_proba_xgb=xgb_model.predict_proba(x_test)
print('ACC:',t)

#print('AUC:',metrics.roc_auc_score(y_test,pred_proba_xgb[:,1]))#验证集上的auc值
#print('F1:',metrics.f1_score(y_test,pre_Xgb))
#print('Recall:',metrics.recall_score(y_test,pre_Xgb))

print(metrics.confusion_matrix(y_test,pre_Xgb))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,pre_Xgb)))

#print('ACC:',accuracy_score(y_test,pre_Xgb))
#xgb.plot_importance(xgb_model)
#plt.show()
print('Finish!')
