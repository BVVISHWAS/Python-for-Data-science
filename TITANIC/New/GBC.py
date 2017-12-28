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
from xgboost import regressor

Data = 'D:\\ML\\TITANIC\\train.csv'
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
 
dataset = dataset.drop('Ticket', axis=1)

dataset =  dataset.drop('Cabin', axis=1)

X = dataset.iloc[:,1 : ]

Y = dataset.iloc[:,0]


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#from sklearn.decomposition import PCA
#pca = PCA(n_components = None)
#X = pca.fit_transform(X)
#explained_variance = pca.explained_variance_ratio_

from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_


#from sklearn.feature_selection import RFE
#model = GradientBoostingClassifier()
#rfe = RFE(model, 4)
#fit = rfe.fit(X, Y)
#print("Num Features: %d" % fit.n_features_)
#print("Selected Features: %s" % fit.support_)
#print("Feature Ranking: %s" % fit.ranking_)
#






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

    
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.1,0.2,0.3], 'kernel' :['rbf','sigmoid','poly']}]
grid_search = GridSearchCV(estimator = SVC(),
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


from sklearn.model_selection import GridSearchCV
parameters = [{'learning_rate': [0.1,0.2,0.3,0.4,0.5], 'max_depth' :[3,4,5,6] ,
'min_samples_leaf' : [1,2,3] , 'min_samples_split' : [100,150,170,180]
,'n_estimators' : [100,150,160,170,180,190] }]
grid_search = GridSearchCV(estimator = GradientBoostingClassifier(),
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#LR: 0.809038 (0.039034)
#LDA: 0.802074 (0.040828)
#KNN: 0.809214 (0.059257)
#CART: 0.800743 (0.059965)
#ADA: 0.807786 (0.054983)
#GBC: 0.826017 (0.051103)
#RFC: 0.797926 (0.048300)
#ETC: 0.802152 (0.051098)
#SVM: 0.831573 (0.054946)

classifier = GradientBoostingClassifier(learning_rate= 0.2,
 max_depth= 3,
 min_samples_leaf= 3,
 min_samples_split= 100,
 n_estimators =160)
classifier.fit(X_train, Y_train)
predicted = classifier.predict(X_validation)
report = classification_report(Y_validation, predicted)
print(report)


##########################################

Data1 = 'D:\\ML\\TITANIC\\test.csv'
dataset1 = read_csv(Data1)

PASSinger = dataset1['PassengerId']

dataset1 = dataset1.drop('PassengerId', axis=1)

dataset1 = dataset1.drop('Name', axis=1)



dataset1['Sex'] = dataset1['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


dataset1['Embarked'] = dataset1['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2 } )

   
dataset1.loc[ dataset1['Age'] <= 16, 'Age'] = 0
dataset1.loc[(dataset1['Age'] > 16) & (dataset1['Age'] <= 32), 'Age'] = 1
dataset1.loc[(dataset1['Age'] > 32) & (dataset1['Age'] <= 48), 'Age'] = 2
dataset1.loc[(dataset1['Age'] > 48) & (dataset1['Age'] <= 64), 'Age'] = 3
dataset1.loc[ dataset1['Age'] > 64, 'Age'] = 4


dataset1['FamilySize'] = dataset1['SibSp'] + dataset1['Parch']
 
dataset1 = dataset1.drop('Ticket', axis=1)

dataset1 =  dataset1.drop('Cabin', axis=1)

X1 = dataset1.iloc[:,1 : ]


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X1)
X1 = imputer.transform(X1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X1 = sc_X.fit_transform(X1)

#from sklearn.decomposition import PCA
#pca = PCA(n_components = None)
#X1 = pca.fit_transform(X1)
#explained_variance = pca.explained_variance_ratio_

from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
X1 = pca.fit_transform(X1)
explained_variance = pca.explained_variance_ratio_


predicted_GBC = classifier.predict(X1)

predicted_GBC = pd.DataFrame(predicted_GBC)

Final_output = pd.concat([PASSinger, predicted_GBC], axis=1)

Final_output.to_csv("D:\\ML\\TITANIC\\New\\Submission_Titanic_GBC.csv", index=False)
