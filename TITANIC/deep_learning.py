# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:19:32 2017

@author: VI279685
"""

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy
# fix random seed for reproducibility
Data = 'D:\\TITANIC\\train.csv'
dataset = read_csv(Data)


dataset = dataset.drop('PassengerId', axis=1)

dataset = dataset.drop('Name', axis=1)



dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


dataset = dataset.drop('Embarked', axis=1)

dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']

dataset.loc[ dataset['FamilySize'] >=1, 'AloneFlag'] = 0

dataset = dataset.drop('FamilySize',axis = 1)
dataset = dataset.drop('SibSp',axis = 1)
dataset = dataset.drop('Parch',axis = 1)
dataset = dataset.drop('Cabin',axis = 1)
dataset = dataset.drop('Ticket', axis=1)


dataset.loc[ dataset['Fare'] <= 16, 'Fare'] = 0
dataset.loc[(dataset['Fare'] > 16) & (dataset['Fare'] <= 32), 'Fare'] = 1
dataset.loc[(dataset['Fare'] > 32) & (dataset['Fare'] <= 48), 'Fare'] = 2
dataset.loc[(dataset['Fare'] > 48) & (dataset['Fare'] <= 64), 'Fare'] = 3
dataset.loc[ dataset['Fare'] > 64, 'Fare'] = 4

dataset['Fare'] = dataset['Fare'].round()


X = dataset.iloc[:,1 : ].values

Y = dataset.iloc[:,0].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

# create model
model = Sequential()
model.add(Dense(12, input_dim=5, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(7, init='uniform', activation='relu'))
model.add(Dense(5, init='uniform', activation='relu'))
model.add(Dense(4, init='uniform', activation='relu'))
model.add(Dense(3, init='uniform', activation='relu'))
model.add(Dense(2, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=20,  verbose=2)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
