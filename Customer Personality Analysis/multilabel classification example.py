# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:00:49 2021

@author: 921000471
"""

import os
os.chdir('D:/Sean/Data Scientist/ds_assignments/NLP')
os.listdir()

import pandas as pd 
import nltk
nltk.download('punkt')
from nltk import punkt
nltk.download('wordnet')
from nltk import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re

from sklearn.preprocessing import LabelEncoder

#load data and check
train = pd.read_csv('disaster_response_messages_training.csv',low_memory=False)
val = pd.read_csv('disaster_response_messages_validation.csv',low_memory=False)
test = pd.read_csv('disaster_response_messages_test.csv',low_memory=False)

#encoding genre columns to be predicted too
genre = LabelEncoder()

train.duplicated().sum() #check for duplicate
train.drop_duplicates(inplace=True) #drop duplicate to avoid redundant training point

train['genre'] = genre.fit_transform(train['genre'])
val['genre'] = genre.transform(val['genre'])
test['genre'] = genre.transform(test['genre'])

#splitting data
xtrain = train['message']
ytrain = train.iloc[:,5:]


xval = val['message']
yval = val.iloc[:,5:]


xtest = test['message']
ytest = test.iloc[:,5:]

#create function for CountVectorizer tokenizer. Use lemmatization rather than stemming

def tokenize(txt):
    teks = re.sub(r"[^a-zA-Z0-9]"," ", txt.lower())
    stpwrd = stopwords.words("english")
    words = word_tokenize(teks)
    lemma = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stpwrd]
    return lemma


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.ensemble import RandomForestClassifier
pipelinerf = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier(RandomForestClassifier(random_state = 42)))
        ])
pipelinerf.fit(xtrain,ytrain)

#define function to show f1 score, precision, recall of each output
from sklearn.metrics import precision_recall_fscore_support

class get_results():
    def __init__(self,ycheck,ypred):
        self.ycheck = ycheck
        self.ypred = ypred
        self.results = pd.DataFrame(columns=['Category', 'f1_score', 'precision', 'recall'])
        num = 0
        for colname in self.ycheck.columns:
            precision, recall, f1_score, support = precision_recall_fscore_support(self.ycheck[colname], self.ypred[:,num], average='weighted',zero_division = 0)
            self.results.at[num+1,'Category'] = colname
            self.results.at[num+1,'f1_score'] = f1_score
            self.results.at[num+1,'precision'] = precision
            self.results.at[num+1,'recall'] = recall
            num += 1
    def summary(self):
        df = self.results[:]
        print('Average f1_score:', df['f1_score'].mean())
        print('Average precision:', df['precision'].mean())
        print('Average recall:', df['recall'].mean())

ypredrf = pipelinerf.predict(xval)
resultrf = get_results(yval,ypredrf)
resultrf.results
resultrf.summary()


#try other model to see better result without tuning
#KNN
from sklearn.neighbors import KNeighborsClassifier
pipelineneighbors = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier(KNeighborsClassifier(weights = 'distance')))
        ])
pipelineneighbors.fit(xtrain,ytrain)

ypredneighbors = pipelineneighbors.predict(xval)
resultneighbors = get_results(yval,ypredneighbors)
resultneighbors.results
resultneighbors.summary()

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
pipelinedt = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier(DecisionTreeClassifier(criterion = 'entropy')))
        ])
pipelinedt.fit(xtrain,ytrain)

ypreddt = pipelinedt.predict(xval)
resultdt = get_results(yval,ypreddt)
resultdt.results
resultdt.summary()

#we'll tune DecisionTree and Random Forest
#from sklearn.model_selection import RandomizedSearchCV

#rf = RandomizedSearchCV(pipelinerf, {'clf__estimator__criterion':['gini','entropy'],
#                                     'clf__estimator__class_weight':['balanced','balanced_subsample',None],
#                                     'clf__estimator__max_features':['auto',None],
#                                     'clf__estimator__random_state' : [42]},n_iter = 5,random_state = 42)
#rf.fit(xtrain, ytrain)
#ypredrftuned = rf.predict(xval)
#resultrftuned = get_results(yval,ypredrftuned)
#resultrftuned.results
#resultrftuned.summary() 

#dt = RandomizedSearchCV(pipelinedt, {'clf__estimator__class_weight':['balanced',None],
#                                     'clf__estimator__criterion':['gini','entropy'],
#                                     'clf__estimator__random_state':[42],
#                                     'clf__estimator__splitter':['best','random'],
#                                     'clf__estimator__max_features':['auto',None]},n_iter = 5,random_state=42)
#dt.fit(xtrain,ytrain)
#ypreddttuned=dt.predict(xval)
#resultdttuned=get_results(yval,ypreddttuned)
#resultdttuned.results
#resultdttuned.summary()

#we skip the tuning part due to time

### Random Forest and Decision Tree actually has almost the same average f1_score
###We'll use Random Forest

modeltesting = pipelinerf.predict(xtest)

from sklearn.metrics import accuracy_score, f1_score

final = get_results(ytest,modeltesting)
final.results
final.summary()

#the model works out okay