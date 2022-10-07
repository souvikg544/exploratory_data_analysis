# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 19:01:31 2021

@author: HP
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from  sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import datetime
dataset=pd.read_csv('train_data.csv',index_col="date",parse_dates=True)
y=pd.read_csv("train_target.csv",index_col='Id',parse_dates=True)
numeric_variables=list(dataset.dtypes[dataset.dtypes !="object"].index)
dataset[numeric_variables].head()
print(dataset.head())
print(dataset.isna().any())
print(dataset.info())
rolling=dataset.rolling(7).mean().head(20)
print(dataset['open'].plot(figsize=(16,6)))
print(dataset.rolling(window=30).mean()['close'].plot())
dataset['close:30 day mean']=dataset['close'].rolling(window=30).mean()
print(dataset[['close','close:30 day mean']].plot(figsize=(16,6)))
print(dataset['close'].expanding(min_periods=1).mean().plot(figsize=(16,6)))
training_set=dataset['open']
training_set=pd.DataFrame(training_set)
dataset.isna().any()
model=RandomForestClassifier(n_estimators=100)
model.fit(dataset[numeric_variables],y)
print("target",accuracy_score(y,model.predict(dataset[numeric_variables])))
test=pd.read_csv("test_data.csv")
print(test[numeric_variables].head())
test['close'].fillna(test.close.mean(),inplace=True)
test=test[numeric_variables].fillna(test.mean()).copy()
y_pred=model.predict(test[numeric_variables])
submission=pd.DataFrame({'Id':test['Id'],'target':y_pred})
submission.to_csv('s2.csv',index=False)
# print(submission)
