# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:19:32 2017

@author: VI279685
"""

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import pandas as pd
import numpy
# fix random seed for reproducibility
Data = 'D:\\ML\\TITANIC\\train.csv'
dataset = pd.read_csv(Data)


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

dataset = dataset.drop('Cabin',axis = 1)
dataset = dataset.drop('Ticket', axis=1)


dataset['Fare'] = dataset['Fare'].round()


X = dataset.iloc[:,1 : ].values

Y = dataset.iloc[:,0].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# create model
model = Sequential()
model.add(Dense(20, input_dim=7, init='uniform', activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(15, init='glorot_uniform', activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(9, init='glorot_uniform', activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(5, init='glorot_uniform', activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(4, init='glorot_uniform', activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(3, init='glorot_uniform', activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(2, init='glorot_uniform', activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, init='glorot_uniform', activation='sigmoid'))
model.add(Dropout(0.2))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=200, batch_size=20)




Data1 = 'D:\\ML\\TITANIC\\test.csv'
dataset1 = pd.read_csv(Data1)

PASSinger = dataset1['PassengerId']

dataset1 = dataset1.drop('PassengerId', axis=1)

dataset1 = dataset1.drop('Name', axis=1)



dataset1['Sex'] = dataset1['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


dataset1 = dataset1.drop('Embarked', axis=1)

dataset1.loc[ dataset1['Age'] <= 16, 'Age'] = 0
dataset1.loc[(dataset1['Age'] > 16) & (dataset1['Age'] <= 32), 'Age'] = 1
dataset1.loc[(dataset1['Age'] > 32) & (dataset1['Age'] <= 48), 'Age'] = 2
dataset1.loc[(dataset1['Age'] > 48) & (dataset1['Age'] <= 64), 'Age'] = 3
dataset1.loc[ dataset1['Age'] > 64, 'Age'] = 4


dataset1['FamilySize'] = dataset1['SibSp'] + dataset1['Parch']

dataset1 = dataset1.drop('Cabin',axis = 1)
dataset1 = dataset1.drop('Ticket', axis=1)


dataset1['Fare'] = dataset1['Fare'].round()


X1 = dataset1.iloc[: : ].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X1)
X1 = imputer.transform(X1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X1 = sc_X.fit_transform(X1)


y_pred = model.predict(X1)
y_pred = (y_pred > 0.5)


y_pred = pd.DataFrame(y_pred)

y_pred['Prediction'] = y_pred


 y_pred['Prediction'] = y_pred['Prediction'].map({False:0,True:1})

Final_output = pd.concat([PASSinger, y_pred['Prediction']], axis=1)

Final_output.to_csv("D:\\ML\\TITANIC\\New\\Submission_Titanic_NN.csv", index=False)

