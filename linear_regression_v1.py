# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:37:41 2019

@author: Charl
"""

import pandas as pd

#**********************************Train & Test dataset**************************************
train = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/train_V2.csv')
train.head()

test = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/test_V2.csv')
test.head()

train.isnull().sum().sum()
test.isnull().sum().sum()

train.winPlacePerc.fillna(1,inplace=True)
train.loc[train['winPlacePerc'].isnull()]

train["distance"] = train["rideDistance"]+train["walkDistance"]+train["swimDistance"]
train["skill"] = train["headshotKills"]+train["roadKills"]
train.drop(['rideDistance','walkDistance','swimDistance','headshotKills','roadKills'],inplace=True,axis=1)
print(train.shape)
train.head()

test["distance"] = test["rideDistance"]+test["walkDistance"]+test["swimDistance"]
test["skill"] = test["headshotKills"]+test["roadKills"]
test.drop(['rideDistance','walkDistance','swimDistance','headshotKills','roadKills'],inplace=True,axis=1)
print(test.shape)
test.head()

predictors = ["kills",
              "killStreaks",
              "killPlace",
              "maxPlace",
              "numGroups",
              "distance",
              "boosts",
              "weaponsAcquired",
              "DBNOs",
               ]

X = train[predictors]
X.head()

y = train['winPlacePerc']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#**********************************Train & Test dataset**************************************

#Build models ...
from sklearn.linear_model import LinearRegression

lm = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=8)
lm.fit(X,y)

predictions = lm.predict(X_test)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error:")
print(mean_absolute_error(y_test, predictions))

#**********************************Submission**************************************
test_id = test["Id"]
submit = pd.DataFrame({'Id': test_id, "winPlacePerc": y_test} , columns=['Id', 'winPlacePerc'])
print(submit.head())

submit.to_csv("submission.csv", index = False)
#**********************************Submission**************************************


#mea=0.1 before feature selection



















