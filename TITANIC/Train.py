# Load libraries
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

Data = 'D:\\TITANIC\\train.csv'
dataset = read_csv(Data)


dataset = dataset.drop('PassengerId', axis=1)

dataset = dataset.drop('Name', axis=1)



dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2 } )

   
dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
 
dataset.loc[ dataset['FamilySize'] <1 ,'FamilySize'] = 1
dataset.loc[ dataset['FamilySize'] >= 1 ,'FamilySize'] = 0

dataset1 = dataset['Ticket'].str.extract('(\s\d+)') 

dataset2 =  dataset['Ticket'].str.extract('(\d*)') 


dataset_ticket = pd.concat([dataset1, dataset2], axis=1)

dataset_ticket = dataset1.combine_first(dataset2)

dataset_ticket = dataset_ticket.str.strip()

dataset_ticket = dataset_ticket.replace(r'\s+',np.nan,regex=True).replace('',np.nan)


dataset = dataset.drop('Ticket', axis=1)


dataset = pd.concat([dataset, dataset_ticket], axis=1)


dataset =  dataset.drop('Cabin', axis=1)

dataset =  dataset.drop('SibSp', axis=1)
dataset =  dataset.drop('Parch', axis=1)

dataset['Fare'] = dataset['Fare'].round()



#from sklearn.feature_selection import RFE
#model = GradientBoostingClassifier()
#rfe = RFE(model, 4)
#fit = rfe.fit(X, Y)
#print("Num Features: %d" % fit.n_features_)
#print("Selected Features: %s" % fit.support_)
#print("Feature Ranking: %s" % fit.ranking_)
#


X = dataset.iloc[:,1 : ].values

Y = dataset.iloc[:,0].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X)
X = scaler.transform(X)


num_folds = 10
seed = 7
scoring = 'accuracy'
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('ADA', AdaBoostClassifier()))
models.append(('GBC', GradientBoostingClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('ETC', ExtraTreesClassifier()))
models.append(('SVM', SVC()))
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
pipelines.append(('ScaledADA', Pipeline([('Scaler', StandardScaler()),('SVM', AdaBoostClassifier())])))
pipelines.append(('ScaledGBC', Pipeline([('Scaler', StandardScaler()),('SVM', GradientBoostingClassifier())])))
pipelines.append(('ScaledRFC', Pipeline([('Scaler', StandardScaler()),('SVM', RandomForestClassifier())])))
pipelines.append(('ScaledETC', Pipeline([('Scaler', StandardScaler()),('SVM', ExtraTreesClassifier())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#accuracies.mean()
#accuracies.std()

from sklearn.model_selection import GridSearchCV
parameters = [{'learning_rate': [0.1,0.2,0.3,0.4,0.5], 'max_depth' :[3,4,5,6] ,
'min_samples_leaf' : [1,2,3] , 'min_samples_split' : [100,150,170,180]
,'n_estimators' : [100,150,160,170,180,190] }]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = 3)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


import pickle
filename = 'D:\\TITANIC\\Titanic.sav'
pickle.dump(classifier, open(filename, 'wb'))


test = 'D:\\TITANIC\\test.csv'
dataset_pred = read_csv(test)

#dataset_pred = dataset_pred.drop('PassengerId', axis=1)

dataset_pred = dataset_pred.drop('Name', axis=1)


dataset_pred = dataset_pred['Ticket'].str.extract('(\s\d+)') 

dataset_pred =  dataset_pred['Ticket'].str.extract('(\d*)') 


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


#dataset = dataset.replace(r'^\s+$', np.nan, regex=True, inplace = True)




#from sklearn.model_selection import GridSearchCV
#parameters = [{'learning_rate': [0.1,0.2,0.3,0.4], 'max_depth' :[3,4,5,6,7,8] ,
#'min_samples_leaf' : [1,2,3,4,5,6] , 'min_samples_split' : [2,3,4,5,6,7,8],
#    'n_estimators' : [100,150,200,250,300] }]
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 10,
#                           n_jobs = 3)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_


#
#Data_pred = 'D:\\TITANIC\\test.csv'
#dataset_pred = read_csv(Data_pred)
#
#
#dataset_pred = dataset_pred.drop('PassengerId', axis=1)
#
#dataset_pred = dataset_pred.drop('Name', axis=1)
#
#
#
#dataset_pred = dataset_pred.drop('Ticket', axis=1)
#
#dataset_pred =  dataset_pred.drop('Cabin', axis=1)
#
#dataset_pred = pd.get_dummies(dataset_pred.Sex, prefix='Sex')
#
#dataset_pred = pd.concat([dataset_pred, dataset_sex], axis=1)
#
#
#dataset_pred = dataset_pred.drop('Sex', axis=1)
#
#dataset_pred = dataset_pred.drop('Sex_female', axis=1)
#
#dataset_Embarked = pd.get_dummies(dataset_pred.Embarked, prefix='Embarked')
#
#dataset_pred = pd.concat([dataset_pred, dataset_Embarked], axis=1)
#
#dataset_pred =  dataset_pred.drop('Embarked', axis=1)
#
#X_new = dataset_pred.iloc[:,1 : -3].values
#
#Y_new = dataset_pred.iloc[:,0].values
#
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X_new)
#X_new = imputer.transform(X_new)
#
#
#
