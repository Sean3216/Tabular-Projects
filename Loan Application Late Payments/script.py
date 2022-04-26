# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 08:41:28 2021

@author: 921000471
"""

import os

os.getcwd()
os.chdir('D:/Sean/Data Scientist/Cermati')
os.listdir()

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np

train = pd.read_csv('app_train.csv')
train.drop('Unnamed: 0',axis = 1,inplace=True)
test = pd.read_csv('app_test.csv')
test.drop('Unnamed: 0',axis = 1,inplace=True)
prev = pd.read_csv('prev_app.csv')
prev.drop('Unnamed: 0',axis = 1,inplace=True)
behavior = pd.read_csv('installment_payment.csv')
behavior.drop('Unnamed: 0',axis = 1,inplace=True)

#bikinkan variabel baru, nanti kita buat 
import matplotlib.pyplot as plt
import seaborn as sns

behavior.drop('LN_ID',axis= 1,inplace=True)
behavior.describe()
behaviorengineered = behavior[:]
behaviorengineered['PREV_PAY_DEFICIT']=behaviorengineered.AMT_INST - behaviorengineered.AMT_PAY
behaviorengineered['PREV_LATENESS'] = behaviorengineered.INST_DAYS - behaviorengineered.PAY_DAYS
behaviorengineered.drop(['INST_DAYS','PAY_DAYS','AMT_INST','AMT_PAY'],axis = 1, inplace=True)

behaviorengineered.fillna(behaviorengineered.median(),inplace=True)


#join by left, banyak variabel yang null karena banyak id yang tidak diketahui past paymentnya
#join by inner aja karena fokus kita adalah data yang sudah berpola. Dari sini nanti kita bisa coba clusterkan
#habis clusterkan, kita join ke train data. Setelah di join ke train data, kita impute yang missing dengan knnimputer
prev.CONTRACT_TYPE.value_counts()
prev.CONTRACT_TYPE.isnull().sum()
#prevbhv = pd.merge(prev,behaviorengineered,'inner',on='SK_ID_PREV') #skip this
prevpaydeficit = behaviorengineered.groupby('SK_ID_PREV')['PREV_PAY_DEFICIT'].agg(lambda x:x.median() if x.notnull().any() else np.nan)
prevlateness = behaviorengineered.groupby('SK_ID_PREV')['PREV_LATENESS'].agg(lambda x:x.median() if x.notnull().any() else np.nan)

prev['PREV_PAY_DEFICIT'] = prev['SK_ID_PREV'].apply(lambda x: prevpaydeficit[x] if x in prevpaydeficit.index else np.nan)
prev['PREV_LATENESS'] = prev['SK_ID_PREV'].apply(lambda x:prevlateness[x] if x in prevlateness.index else np.nan)

prev.PREV_PAY_DEFICIT.isnull().sum()/len(prev) #it's so large, near to half of the data
prev.PREV_LATENESS.isnull().sum()/len(prev) 

#we don't drop null because we want tp join the data with train data then impute what's missing with knn imputer

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

#kalo di prevbhv, yang categorical ada CONTRACT_TYPE,WEEKDAYS_APPLY, CONTRACT_STATUS, sama YIELD_GROUP
LEcontract_type = LabelEncoder()
LEweekdays_apply = LabelEncoder()
LEcontract_status = LabelEncoder()
LEyield_group = LabelEncoder()

prev['CONTRACT_TYPE'] = LEcontract_type.fit_transform(prev.CONTRACT_TYPE)
prev.WEEKDAYS_APPLY = LEweekdays_apply.fit_transform(prev.WEEKDAYS_APPLY)
prev.CONTRACT_STATUS = LEcontract_status.fit_transform(prev.CONTRACT_STATUS)
prev.YIELD_GROUP = LEyield_group.fit_transform(prev.YIELD_GROUP)
prev.describe()
prev.drop(['SK_ID_PREV'],axis = 1, inplace = True)

#lihat variable prevbhv mana yang saling berkorelasi untuk dibuang salah satu
corrprevbhv = prev.corr()
corr_triuprevbhv = corrprevbhv.where(~np.tril(np.ones(corrprevbhv.shape)).astype(np.bool))
corr_triuprevbhv = corr_triuprevbhv.stack()
corr_triuprevbhv.name = 'Pearson Correlation Coefficient'
corr_triuprevbhv.index.names = ['First Var', 'Second Var']
corr_triuprevbhv[(corr_triuprevbhv > 0.3)|(corr_triuprevbhv < -0.3)].to_frame()
#dari Correlation, kita ambil beberapa variabel saja yang sekiranya bisa mewakilkan variabel lain
prevbhvfinal = prev[['LN_ID','CONTRACT_TYPE','CONTRACT_STATUS','AMT_DOWN_PAYMENT','PRICE','WEEKDAYS_APPLY','HOUR_APPLY',
                        'DAYS_DECISION','PREV_PAY_DEFICIT','PREV_LATENESS','TERMINATION']]
#dimension reduction before clustering
#from sklearn.preprocessing import StandardScaler
#scalePCA = StandardScaler()
#prevbhvscalePCA = pd.DataFrame(scalePCA.fit_transform(prevbhvfinal),columns=prevbhvfinal.columns)
#prevbhvFA = prevbhvscalePCA.dropna()
#from factor_analyzer import FactorAnalyzer
#from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
#chi_square_value,p_value=calculate_bartlett_sphericity(prevbhvFA)
#chi_square_value, p_value

#from factor_analyzer.factor_analyzer import calculate_kmo
#kmo_all,kmo_model=calculate_kmo(prevbhvFA)
#kmo_model

#prevbhvfinal ada ID yang nilainya lebih dari 1 kali record muncul, maka kita hrs groupingkan
prevbhvfinal.info()
prevbhvfinal.LN_ID.value_counts()

from scipy.stats import mode

sns.pairplot(prevbhvfinal)
prevbhvfinal.AMT_DOWN_PAYMENT.describe()
prevbhvfinal.PRICE.describe()
prevbhvfinal.HOUR_APPLY.describe()
prevbhvfinal.DAYS_DECISION.describe()
prevbhvfinal.PREV_PAY_DEFICIT.describe()
prevbhvfinal.PREV_LATENESS.describe()
prevbhvfinal.TERMINATION.describe()

contract_type = prevbhvfinal.groupby(['LN_ID'])['CONTRACT_TYPE'].agg(lambda x: mode(x)[0][0] if x.notnull().any() else np.nan )
contract_status = prevbhvfinal.groupby(['LN_ID'])['CONTRACT_STATUS'].agg(lambda x: mode(x)[0][0] if x.notnull().any() else np.nan)
amt_down_payment = prevbhvfinal.groupby(['LN_ID'])['AMT_DOWN_PAYMENT'].agg(lambda x: x.median() if x.notnull().any() else np.nan )
price = prevbhvfinal.groupby(['LN_ID'])['PRICE'].agg(lambda x: x.median() if x.notnull().any() else np.nan )
weekdays_apply = prevbhvfinal.groupby(['LN_ID'])['WEEKDAYS_APPLY'].agg(lambda x: mode(x)[0][0] if x.notnull().any() else np.nan )
hour_apply = prevbhvfinal.groupby(['LN_ID'])['HOUR_APPLY'].agg(lambda x: x.median() if x.notnull().any() else np.nan )
days_decision = prevbhvfinal.groupby(['LN_ID'])['DAYS_DECISION'].agg(lambda x: x.median() if x.notnull().any() else np.nan )
prev_pay_deficit = prevbhvfinal.groupby(['LN_ID'])['PREV_PAY_DEFICIT'].agg(lambda x: x.median() if x.notnull().any() else np.nan )
prev_lateness = prevbhvfinal.groupby(['LN_ID'])['PREV_LATENESS'].agg(lambda x: x.median() if x.notnull().any() else np.nan )
termination = prevbhvfinal.groupby(['LN_ID'])['TERMINATION'].agg(lambda x: x.median() if x.notnull().any() else np.nan )

#combine to train
train.info()
train.GENDER.value_counts()#dummy var (without drop first)
test.GENDER.value_counts()

train.INCOME_TYPE.value_counts()#label encoder
test.INCOME_TYPE.value_counts()

train.EDUCATION.value_counts()#ordinal encoder
test.EDUCATION.value_counts()

train.FAMILY_STATUS.value_counts()#label encoder
test.FAMILY_STATUS.value_counts()

train.HOUSING_TYPE.value_counts()#label encoder
test.HOUSING_TYPE.value_counts()

train.ORGANIZATION_TYPE.value_counts()#label encoder
test.ORGANIZATION_TYPE.value_counts()

LEincometype = LabelEncoder()
LEeducation = OrdinalEncoder(categories = [['Academic degree','Lower secondary','Secondary / secondary special',
                                           'Incomplete higher','Higher education']])
LEfamilystatus = LabelEncoder()
LEhousingtypes = LabelEncoder()
LEorganizationtype = LabelEncoder()


train['PREV_CONTRACT_TYPE'] = train['LN_ID'].apply(lambda x: contract_type[x] if x in contract_type.index else np.nan)
train['PREV_AMT_DOWN_PAYMENT'] = train['LN_ID'].apply(lambda x: amt_down_payment[x] if x in amt_down_payment.index else np.nan)
train['PREV_PRICE'] = train['LN_ID'].apply(lambda x: price[x] if x in price.index else np.nan)
train['PREV_WEEKDAYS_APPLY'] = train['LN_ID'].apply(lambda x: weekdays_apply[x] if x in weekdays_apply.index else np.nan)
train['PREV_HOUR_APPLY'] = train['LN_ID'].apply(lambda x: hour_apply[x] if x in hour_apply.index else np.nan)
train['PREV_DAYS_DECISION'] = train['LN_ID'].apply(lambda x: days_decision[x] if x in days_decision.index else np.nan)
train['PREV_PAY_DEFICIT'] = train['LN_ID'].apply(lambda x: prev_pay_deficit[x] if x in prev_pay_deficit.index else np.nan)
train['PREV_LATENESS'] = train['LN_ID'].apply(lambda x: prev_lateness[x] if x in prev_lateness.index else np.nan)
train['PREV_TERMINATION'] = train['LN_ID'].apply(lambda x: termination[x] if x in termination.index else np.nan)
train['PREV_CONTRACT_STATUS'] = train['LN_ID'].apply(lambda x:contract_status[x] if x in contract_status.index else np.nan)

fortraingenddummy = pd.get_dummies(train.GENDER)
train['GENDER_F'], train['GENDER_M'] = fortraingenddummy['F'],fortraingenddummy['M']
train.drop('GENDER',axis = 1, inplace= True)
train['INCOME_TYPE'] = LEincometype.fit_transform(train['INCOME_TYPE'])
train['EDUCATION'] = LEeducation.fit_transform(train.loc[:,['EDUCATION']])
train['FAMILY_STATUS'] = LEfamilystatus.fit_transform(train['FAMILY_STATUS'])
train['HOUSING_TYPE'] = LEhousingtypes.fit_transform(train['HOUSING_TYPE'])
train['ORGANIZATION_TYPE'] = LEorganizationtype.fit_transform(train['ORGANIZATION_TYPE'])

train['CONTRACT_TYPE'] = LEcontract_type.transform(train['CONTRACT_TYPE'])
train['WEEKDAYS_APPLY'] = LEweekdays_apply.transform(train['WEEKDAYS_APPLY'])

test['PREV_CONTRACT_TYPE'] = test['LN_ID'].apply(lambda x: contract_type[x] if x in contract_type.index else np.nan)
test['PREV_AMT_DOWN_PAYMENT'] = test['LN_ID'].apply(lambda x: amt_down_payment[x] if x in amt_down_payment.index else np.nan)
test['PREV_PRICE'] = test['LN_ID'].apply(lambda x: price[x] if x in price.index else np.nan)
test['PREV_WEEKDAYS_APPLY'] = test['LN_ID'].apply(lambda x: weekdays_apply[x] if x in weekdays_apply.index else np.nan)
test['PREV_HOUR_APPLY'] = test['LN_ID'].apply(lambda x: hour_apply[x] if x in hour_apply.index else np.nan)
test['PREV_DAYS_DECISION'] = test['LN_ID'].apply(lambda x: days_decision[x] if x in days_decision.index else np.nan)
test['PREV_PAY_DEFICIT'] = test['LN_ID'].apply(lambda x: prev_pay_deficit[x] if x in prev_pay_deficit.index else np.nan)
test['PREV_LATENESS'] = test['LN_ID'].apply(lambda x: prev_lateness[x] if x in prev_lateness.index else np.nan)
test['PREV_TERMINATION'] = test['LN_ID'].apply(lambda x: termination[x] if x in termination.index else np.nan)
test['PREV_CONTRACT_STATUS'] = test['LN_ID'].apply(lambda x:contract_status[x] if x in contract_status.index else np.nan)

fortestgenddummy = pd.get_dummies(test.GENDER)
test['GENDER_F'], test['GENDER_M'] = fortestgenddummy['F'],fortestgenddummy['M']
test.drop('GENDER',axis = 1, inplace= True)
test['INCOME_TYPE'] = LEincometype.transform(test['INCOME_TYPE'])
test['EDUCATION'] = LEeducation.transform(test.loc[:,['EDUCATION']])
test['FAMILY_STATUS'] = LEfamilystatus.transform(test['FAMILY_STATUS'])
test['HOUSING_TYPE'] = LEhousingtypes.transform(test['HOUSING_TYPE'])
test['ORGANIZATION_TYPE'] = LEorganizationtype.transform(test['ORGANIZATION_TYPE'])

test['CONTRACT_TYPE'] = LEcontract_type.transform(test['CONTRACT_TYPE'])
test['WEEKDAYS_APPLY'] = LEweekdays_apply.transform(test['WEEKDAYS_APPLY'])

#training and testing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

train.isnull().sum()/len(train)
test.isnull().sum()/len(test)

scale = StandardScaler()
impute=KNNImputer()

train.drop('EXT_SCORE_1',axis = 1, inplace = True)
test.drop('EXT_SCORE_1',axis = 1, inplace=True)
#checkclassimbalance
train.TARGET.value_counts()

xtrain = train.drop('TARGET',axis = 1, inplace = False)
ytrain = train['TARGET']

xtest = test.drop('TARGET',axis = 1, inplace = False)
ytest = test['TARGET']

xtrainscaled = pd.DataFrame(scale.fit_transform(xtrain),columns = xtrain.columns)
xtrainscaledimpute = pd.DataFrame(impute.fit_transform(xtrainscaled),columns = xtrainscaled.columns)

xtestscaled = pd.DataFrame(scale.transform(xtest),columns=xtest.columns)

xtestscaledimpute = pd.DataFrame(impute.fit_transform(xtestscaled),columns = xtestscaled.columns)

#check correlation after scale and impute
corr = train.corr()
corr_triu = corr.where(~np.tril(np.ones(corr.shape)).astype(np.bool))
corr_triu = corr_triu.stack()
corr_triu.name = 'Pearson Correlation Coefficient'
corr_triu.index.names = ['First Var', 'Second Var']
corr_triu[(corr_triu > 0.3) | (corr_triu < -0.3)].to_frame()
#drop variables that can be describe by other variable
#drop price and days_work
xtrainscaledimpute.drop(['PRICE','DAYS_WORK'],axis = 1 , inplace = True)
xtestscaledimpute.drop(['PRICE','DAYS_WORK'], axis = 1, inplace = True)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42)
xtrainfinal, ytrainfinal = sm.fit_resample(xtrainscaledimpute, ytrain)


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score,f1_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


#logreg
logreg = GridSearchCV(LogisticRegression(max_iter=300),dict(solver = ['newton-cg', 'lbfgs', 'sag', 'saga']),
                      scoring='roc_auc')
logreg.fit(xtrainfinal,ytrainfinal)
logreg.best_estimator_
logregmodel = LogisticRegression(max_iter=300, solver='sag')
logregmodel.fit(xtrainfinal,ytrainfinal)
ypredlogreg = (logregmodel.predict_proba(xtestscaledimpute)[:,1]>=0.62).astype(int)
rocauclogreg = round(roc_auc_score(ytest,ypredlogreg),3)
classreportlogreg = classification_report(ytest,ypredlogreg)
print('Logistic Regression Classification Report\n'+classreportlogreg+"\nROC AUC Score: "+str(rocauclogreg)+"\nF1-Score: "+str(f1_score(ytest,ypredlogreg)))
acclogreg = round(accuracy_score(ytest,ypredlogreg),3)

#Perceptron
Perceptron = GridSearchCV(Perceptron(random_state = 42),dict(penalty=['l2','l1','elasticnet','None'],
                                                             class_weight = ['balanced','None']))
Perceptron.fit(xtrainfinal,ytrainfinal)
ypredPercept = Perceptron.predict(xtestscaledimpute)
rocaucPercept = round(roc_auc_score(ytest,ypredPercept),3)
classreportPercept = classification_report(ytest,ypredPercept)
print('Perceptron Classification Report\n'+classreportPercept+"\nROC AUC Score: "+str(rocaucPercept)+"\nF1-Score: "+str(f1_score(ytest,ypredPercept)))
accPercept= round(accuracy_score(ytest,ypredPercept),3)

#RandomForestClassifier
RF = RandomizedSearchCV(RandomForestClassifier(random_state=42),dict(n_estimators=[100,150,200],
                                                criterion = ['gini','entropy'],
                                                max_features = ['sqrt','log2']),
                        random_state=42,scoring='roc_auc')
RF.fit(xtrainfinal,ytrainfinal)
RF.best_params_
RFmodel = RandomForestClassifier(n_estimators= 200, max_features= 'log2', criterion= 'entropy',random_state=42)
RFmodel.fit(xtrainfinal,ytrainfinal)
ypredrf = (RFmodel.predict_proba(xtestscaledimpute)[:,1]>=0.318).astype(int)
rocaucrf = round(roc_auc_score(ytest,ypredrf),3)
classreportrf = classification_report(ytest,ypredrf)
print('Random Forest Classification Report \n'+classreportrf+"\nROC AUC Score: "+str(rocaucrf)+"\nF1-Score: "+str(f1_score(ytest,ypredrf)))
accrf = round(accuracy_score(ytest,ypredrf),3)

from scipy.stats import norm

def logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    p = (1 - norm.cdf(abs(t))) * 2
    return p

listcol = list(xtrainfinal.columns)
listcol.insert(0,'Intercept')
logregpvalue = pd.Series(logit_pvalue(logregmodel, xtrainfinal),index = listcol)
logregpvalue = round(logregpvalue,3)

logregcoef = pd.Series(logregmodel.coef_[0],index=xtrainfinal.columns)
contracttypeencoded=train.CONTRACT_TYPE.head()
contracttypeori = LEcontract_type.inverse_transform(train.CONTRACT_TYPE.head())

RFFeatureImportances = pd.Series(RFmodel.feature_importances_,index = xtrainfinal.columns)
RFFeatureImportances.sort_values(axis = 0,inplace=True, ascending=False)

from sklearn.feature_selection import RFE
importance = RFE(logregmodel,n_features_to_select = 1)
importance.fit(xtrainfinal,ytrainfinal)

rank = importance.ranking_
feature_ranks = []
for i in rank:
    feature_ranks.append(f"{i}. {xtrainfinal.columns[i-1]}")
feature_ranks

