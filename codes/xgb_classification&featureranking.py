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
'''
conn = pymysql.connect(host='192.168.1.131',user='yichen', password='0000', database='yelp_db', charset='utf8')
cur = conn.cursor()

#query="select id as user_id, review_count,yelping_since,useful,funny,cool,fans,average_stars, compliment_hot+ compliment_more+ compliment_profile+ compliment_cute+ compliment_list+ compliment_note+ compliment_plain+compliment_cool+compliment_funny+compliment_writer+compliment_photos as sum_compliment from user"
query='select id as review_id,business_id,user_id,stars,date,useful,funny,cool from review'
cur.execute(query)
#cur.execute('commit')
t=cur.fetchall()
#user_info_50000=pd.DataFrame(list(t), columns = [i[0] for i in cur.description])
review_info_notext=pd.DataFrame(list(t), columns = [i[0] for i in cur.description])
print('done')
review_info_notext.to_csv('E:/DATA/DATA in CU Boulder/CSCI 5622/review_info_notext.csv')
'''
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
###SVM returns only label
# model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr')
### RandomForest could return the proba
# model = RandomForestClassifier(n_estimators=70, min_samples_split=10, min_samples_leaf=10,max_depth =22,max_features='sqrt' ,random_state=10,class_weight = 'balanced')
###GradientBoosting return the proba
# model = GradientBoostingClassifier()
###adaboost could return the proba
# model = AdaBoostClassifier( base_estimator = RandomForestClassifier(n_estimators=70, min_samples_split=10, min_samples_leaf=10,max_depth =22,max_features='sqrt' ,random_state=10,class_weight = 'balanced'),                     #algorithm="SAMME",
                 #n_estimators=50, learning_rate=0.1)
    ###grid search for Xgb
#find eta
#print('find learning_rate')
#param_test1 = { 'learning_rate':[0.001,0.01,0.05,0.1,0.3,0.5,1.0,2.0,10.0]}
#gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(max_depth = 6, min_child_weight = 1, gamma = 0),
               #param_grid = param_test1, scoring='accuracy',cv=5)
#gsearch1.fit(x_train_Xgb, y_train_Xgb)
#print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
 #find min_child_weight
'''
print('find min_child_weight')
param_test2 = { 'min_child_weight':[0.1,0.5,1.0,1.5,2.0]}
gsearch2 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate = 0.3, max_depth = 6, gamma = 0),
               param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch2.fit(x_train, y_train)
print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
'''    #find max_depth
'''
print('find max_depth')
param_test2 = { 'max_depth':[1,3,6,8,10]}
gsearch2 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate = 0.3, min_child_weight = 1, gamma = 0),
               param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch2.fit(x_train, y_train)
print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
'''    #find gamma
'''
print('find gamma')
param_test2 = { 'gamma':[0,0.1,0.5,1.0]}
gsearch2 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate = 0.3, min_child_weight = 1, max_depth = 6, gamma = 0),
               param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch2.fit(x_train, y_train)
print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
'''    # confusion matrix
'''confusion = confusion_matrix(y_test, predictions)
print("\t" + "\t".join(str(x) for x in range(0, 2)))
print("".join(["-"] * 50))
for ii in range(0, 2):
    jj = ii
    print("%i:\t" % jj + "\t".join(str(confusion[ii][x]) for x in range(0, 2)))    print(pre)
'''
#print(f1_score(y_test, predictions))
#print(precision_score(y_test, predictions))
#print(recall_score(y_test, predictions))
# print(roc_auc_score(y_test, predictions))
# print(classification_report(y_test, predictions))

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
