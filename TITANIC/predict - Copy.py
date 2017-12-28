# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:36:18 2017

@author: VI279685
"""
import pickle
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Imputer


# for dataset in combine:
    # dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

	# for dataset in combine:
    # dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

	# for dataset in combine:    
    # dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    # dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    # dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    # dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    # dataset.loc[ dataset['Age'] > 64, 'Age']

Data = 'D:\\TITANIC\\test.csv'
dataset = read_csv(Data)


dataset = dataset.drop('Name', axis=1)


dataset1 = dataset['Ticket'].str.extract('(\s\d+)') 

dataset2 =  dataset['Ticket'].str.extract('(\d*)') 


dataset_ticket = pd.concat([dataset1, dataset2], axis=1)

dataset_ticket = dataset1.combine_first(dataset2)

dataset_ticket = dataset_ticket.str.strip()

dataset_ticket = dataset_ticket.replace(r'\s+',np.nan,regex=True).replace('',np.nan)


dataset = dataset.drop('Ticket', axis=1)


dataset = pd.concat([dataset, dataset_ticket], axis=1)



dataset_sex = pd.get_dummies(dataset.Sex, prefix='Sex',drop_first=True)

dataset = pd.concat([dataset, dataset_sex], axis=1)


dataset = dataset.drop('Sex', axis=1)


dataset_Embarked = pd.get_dummies(dataset.Embarked, prefix='Embarked',drop_first=True)

dataset = pd.concat([dataset, dataset_Embarked], axis=1)

dataset =  dataset.drop('Embarked', axis=1)

dataset = dataset.rename(columns={'Sex_male': 'Sex'})

dataset = dataset.rename(columns={'Embarked_Q': 'Embarked_1','Embarked_S':'Embarked_2' })

dataset =  dataset.drop('Cabin', axis=1)



dataset =  dataset.drop('SibSp', axis=1)
dataset =  dataset.drop('Parch', axis=1)

dataset =  dataset.drop('Embarked_1', axis=1)

dataset =  dataset.drop('Embarked_2', axis=1)

X = dataset.iloc[:,1 : ].values

passinger_id= dataset['PassengerId']

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X)




loaded_model = pickle.load(open('D:\\TITANIC\\Titanic.sav', 'rb'))
result = loaded_model.predict(X)
print(result)

result = pd.DataFrame(result)

Final_output = pd.concat([passinger_id, result], axis=1)

Final_output.to_csv("D:\\TITANIC\\Submission_Titanic.csv", index=False)