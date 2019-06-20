# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 02:45:20 2019

@author: Charl
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/train_V2.csv')
train.head()

X = train.drop('winPlacePerc', 1)
y = train['winPlacePerc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#from sklearn.preprocessing import LabelEncoder
#LC=LabelEncoder()
#X_train[:,15]=LC.fit_transform(X_train[:,15])

from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True, n_jobs=8)

lreg = model.fit(X_train, y_train)
print("Train Score:", lreg.score(X_train, y_train))
print("Test Score:", lreg.score(X_test, y_test))

test = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/test_V2.csv')
test.head()

pred = lreg.predict(test)
submission = pd.DataFrame.from_dict(data={'Id': test['Id'], 'winPlacePerc': pred})

submission.head()
submission.to_csv('submission.csv', index=False)