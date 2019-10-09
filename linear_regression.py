# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:37:41 2019

@author: Charl
"""

#%% Import data

import pandas as pd
import numpy as np
from time import time

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
train['assistsNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['roadKillsNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['vehicleDestroysNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['killPointsNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['headshotKillsNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['revivesNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)

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

#%% Hyperparameter tuning

from sklearn.model_selection import GridSearchCV

def log(x):
    # can be used to write to a log file
    print(x)

# Utility function to report best scores (from scikit-learn.org)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            log("Model with rank: {0}".format(i))
            log("Mean validation score: {0:.5f} (std: {1:.5f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            log("Parameters: {0}".format(results['params'][candidate]))
            log("")

# Function to determine the best fit (from scikit-learn.org)
def best_fit(clf, X_train, y_train):
    
    param_grid = {
                    'max_features':['sqrt','log2',None],
                    'max_depth': np.arange(1, 15),
                    'min_samples_split': range(2,16,2),
                    'min_samples_leaf': range(2,20,2),
                    'max_leaf_nodes': [5,10,None],
                 }

    # Run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=8)

    import time as ttt
    now = time()
    log(ttt.ctime())
    
    grid_search.fit(X_train, y_train)
    
    report(grid_search.cv_results_, n_top=10)
    
    log(100*"-")
    log(ttt.ctime())
    log("Search (5-fold cross validation) took %.5f seconds for %d candidate parameter settings."
        % (time() - now, len(grid_search.cv_results_['params'])))
    log('')
    log("The best parameters are %s with a score of %0.5f"
        % (grid_search.best_params_, grid_search.best_score_))
    
    return grid_search

#%% Model

from sklearn.linear_model import LinearRegression

LR = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=8)
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

#%% Submission

train_id = train["Id"]
submit = pd.DataFrame({'Id': train_id, "winPlacePerc": y_test} , columns=['Id', 'winPlacePerc'])
print("Submission head\n {}".format(submit.head()))

submit.to_csv("submission.csv", index = False)

#%%

#mea=0.1 before feature selection
