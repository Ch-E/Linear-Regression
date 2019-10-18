# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:37:41 2019

@author: Charl
"""

#%% Import data

import pandas as pd

#%% Read dataset

train = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/train_V2.csv')

train.winPlacePerc.fillna(1,inplace=True)
train.loc[train['winPlacePerc'].isnull()]

# Create distance feature
train["distance"] = train["rideDistance"]+train["walkDistance"]+train["swimDistance"]
train.drop(['rideDistance','walkDistance','swimDistance'],inplace=True,axis=1)

# Create headshot_rate feature
train['headshot_rate'] = train['headshotKills'] / train['kills']
train['headshot_rate'] = train['headshot_rate'].fillna(0)

# Create playersJoined feature - used for normalisation
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')

#%% Data cleaning - removing outliers

# Row with NaN 'winPlacePerc' value - pointed out by averagemn (https://www.kaggle.com/donkeys)
train.drop(2744604, inplace=True)

# Players who got kills without moving
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['distance'] == 0))
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)

# Players who got more than 10 roadkills
train.drop(train[train['roadKills'] > 10].index, inplace=True)

# Players who got more than 30 kills
train[train['kills'] > 30].head(10)

# Players who made a minimum of 9 kills and have a headshot_rate of 100%
train[(train['headshot_rate'] == 1) & (train['kills'] > 8)].head(10)

# Players who made kills with a distance of more than 1 km
train.drop(train[train['longestKill'] >= 1000].index, inplace=True)

# Players who acquired more than 80 weapons
train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)

# Players how use more than 40 heals
train['heals'] = train['boosts']+train['heals']
train.drop(train[train['heals'] >= 40].index, inplace=True)

# Create normalised features
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)
train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['assistsNorm'] = train['assists']*((100-train['playersJoined'])/100 + 1)
train['roadKillsNorm'] = train['roadKills']*((100-train['playersJoined'])/100 + 1)
train['vehicleDestroysNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['killPointsNorm'] = train['vehicleDestroys']*((100-train['playersJoined'])/100 + 1)
train['headshotKillsNorm'] = train['headshotKills']*((100-train['playersJoined'])/100 + 1)
train['revivesNorm'] = train['revives']*((100-train['playersJoined'])/100 + 1)


#%%

# Features that will be used for training
predictors = [
              "numGroups",
              "distance",
              "boosts",
              "killStreaks",
              "DBNOs",
              "killPlace",
              "killStreaks",
              "longestKill",
              "heals",
              "weaponsAcquired",
              "headshot_rate",
              "assistsNorm",
              "headshotKillsNorm",
              "damageDealtNorm",
              "killPointsNorm",
              "revivesNorm",
              "roadKillsNorm",
              "vehicleDestroysNorm",
              "killsNorm",
              "maxPlaceNorm",
              "matchDurationNorm",
              ]

X = train[predictors]
X.head()

y = train['winPlacePerc']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#%% Model

from sklearn.linear_model import LinearRegression

LR = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)
LR.fit(X_train, y_train)

predictions = LR.predict(X_test)

#%% Evaluation

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

MAE = mean_absolute_error(y_test, predictions)
MSE = mean_squared_error(y_test, predictions)
R2 = r2_score(y_test, predictions)

print("Metrics:")
print("-------------------------------")
print("Mean Absolute Error: {}".format(MAE))
print("Mean Squared Error: {}".format(MSE))
print("R2 Score: {}".format(R2))

# Cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

cross_val_prediction = cross_val_predict(LR, X_train, y_train, cv=5)

print("\n---------------------------------")
print("5-FOLD CROSS-VALIDATION")
print("---------------------------------")
print("Cross-validation score (R2): {}".format(cross_val_score(LR, X_train, y_train, cv=5)))

#%% Data cleaning (Test set)

test = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/test_V2.csv')

# Create distance feature
test["distance"] = test["rideDistance"]+test["walkDistance"]+test["swimDistance"]
test.drop(['rideDistance','walkDistance','swimDistance'],inplace=True,axis=1)

# Create headshot_rate feature
test['headshot_rate'] = test['headshotKills'] / test['kills']
test['headshot_rate'] = test['headshot_rate'].fillna(0)

# Create playersJoined feature - used for normalisation
test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')

# Players who got kills without moving
test['killsWithoutMoving'] = ((test['kills'] > 0) & (test['distance'] == 0))
test.drop(test[test['killsWithoutMoving'] == True].index, inplace=True)

# Players who got more than 10 roadkills
test.drop(test[test['roadKills'] > 10].index, inplace=True)

# Players who made a minimum of 9 kills and have a headshot_rate of 100%
test[(test['headshot_rate'] == 1) & (test['kills'] > 8)].head(10)

# Players who made kills with a distance of more than 1 km
test.drop(test[test['longestKill'] >= 1000].index, inplace=True)

# Players who acquired more than 80 weapons
test.drop(test[test['weaponsAcquired'] >= 80].index, inplace=True)

# Players how use more than 40 heals
test['heals'] = test['boosts']+test['heals']
test.drop(test[test['heals'] >= 40].index, inplace=True)

# Create normalised features
test['killsNorm'] = test['kills']*((100-test['playersJoined'])/100 + 1)
test['damageDealtNorm'] = test['damageDealt']*((100-test['playersJoined'])/100 + 1)
test['maxPlaceNorm'] = test['maxPlace']*((100-test['playersJoined'])/100 + 1)
test['matchDurationNorm'] = test['matchDuration']*((100-test['playersJoined'])/100 + 1)
test['assistsNorm'] = test['assists']*((100-test['playersJoined'])/100 + 1)
test['roadKillsNorm'] = test['roadKills']*((100-test['playersJoined'])/100 + 1)
test['vehicleDestroysNorm'] = test['matchDuration']*((100-test['playersJoined'])/100 + 1)
test['killPointsNorm'] = test['vehicleDestroys']*((100-test['playersJoined'])/100 + 1)
test['headshotKillsNorm'] = test['headshotKills']*((100-test['playersJoined'])/100 + 1)
test['revivesNorm'] = test['revives']*((100-test['playersJoined'])/100 + 1)

#%% Prediction

x_test = test[predictors]
y_predict = LR.predict(x_test)

y_predict[y_predict > 1] = 1
y_predict[y_predict < 0] = 0.0

test['winPlacePercPredictions'] = y_predict

#%% Submission

aux = test.groupby(['matchId','groupId'])['winPlacePercPredictions'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
test = test.merge(aux, how='left', on=['matchId','groupId'])
submission = test[['Id','winPlacePerc']]

print("Submission head\n {}".format(submission.head()))
submission.to_csv("submission.csv", index=False)

#%%