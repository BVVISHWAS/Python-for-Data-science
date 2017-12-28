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

Data = 'D:\\ML\\Mushroom prediction\\mushrooms.csv'
dataset = read_csv(Data)


dataset['cap-shape'].unique()

dataset['cap-shape'] = dataset['cap-shape'].map({'x':1, 'b':2, 's':3, 'f':4, 'k':5, 'c':6})

dataset['cap-surface'].unique()

dataset['cap-surface'] = dataset['cap-surface'].map({'s':1, 'y':2, 'f':3, 'g':4})


dataset['bruises'].unique()

dataset['bruises'] = dataset['bruises'].map({'t':1, 'f':2})

dataset['cap-color'].unique()

dataset['cap-color'] = dataset['cap-color'].map({'n':1, 'y':2, 'w':3, 'g':4, 'e':5, 'p':6, 'b':7, 'u':8, 'c':9, 'r':10})


dataset['odor'].unique()

dataset['odor'] = dataset['odor'].map({'p':1, 'a':2, 'l':3, 'n':4, 'f':5, 'c':6, 'y':7, 's':8, 'm':9})

dataset['gill-attachment'].unique()

dataset['gill-attachment'] = dataset['gill-attachment'].map({'f':1, 'a':2})


dataset['gill-spacing'].unique()

dataset['gill-spacing'] = dataset['gill-spacing'].map({'c':10, 'w':20})

dataset['gill-size'].unique()

dataset['gill-size'] = dataset['gill-size'].map({'n':100, 'b':200})

dataset['gill-color'].unique()

dataset['gill-color'] = dataset['gill-color'].map({'k':1, 'n':2, 'g':3, 'p':4, 'w':5, 'h':6, 'u':7, 'e':8, 'b':9, 'r':10, 'y':11, 'o':12})

dataset['stalk-shape'].unique()

dataset['stalk-shape'] = dataset['stalk-shape'].map({'e':101, 't':102})

dataset['stalk-root'].unique()

dataset['stalk-root'] = dataset['stalk-root'].map({'e':1, 'c':2, 'b':3, 'r':4, '?':0})

dataset['stalk-root'].value_counts()

dataset['stalk-surface-above-ring'].unique()

dataset['stalk-surface-above-ring'] = dataset['stalk-surface-above-ring'].map({'s':1, 'f':2, 'k':3, 'y':4})


dataset['stalk-surface-below-ring'].unique()

dataset['stalk-surface-below-ring'] = dataset['stalk-surface-below-ring'].map({'s':1, 'f':2, 'y':4, 'k':3})


dataset['stalk-color-above-ring'].unique()

dataset['stalk-color-above-ring'] = dataset['stalk-color-above-ring'].map({'w':1, 'g':2, 'p':3, 'n':4, 'b':5, 'e':6, 'o':7, 'c':8, 'y':9})


dataset['stalk-color-below-ring'].unique()

dataset['stalk-color-below-ring'] = dataset['stalk-color-below-ring'].map({'w':1, 'p':3, 'g':2, 'b':5, 'n':4, 'e':6, 'y':9, 'o':7, 'c':8})


dataset['veil-type'].unique()


dataset.drop('veil-type',axis=1, inplace=True)


dataset['veil-color'].unique()
dataset['veil-color'] = dataset['veil-color'].map({'w':11, 'n':12, 'o':13, 'y':14})

dataset['ring-number'].unique()
dataset['ring-number'] = dataset['ring-number'].map({'o':21, 't':31, 'n':41})


dataset['ring-type'].unique()
dataset['ring-type'] = dataset['ring-type'].map({'p':2, 'e':4, 'l':6, 'f':8, 'n':10})


dataset['spore-print-color'].unique()
dataset['spore-print-color'] = dataset['spore-print-color'].map({'k':1, 'n':2, 'u':3, 'h':4, 'w':5, 'r':6, 'o':7, 'y':8, 'b':9})



dataset['population'].unique()
dataset['population'] = dataset['population'].map({'s':1, 'n':2, 'a':3, 'v':4, 'y':5, 'c':6})

dataset['habitat'].unique()
dataset['habitat'] = dataset['habitat'].map({'u':1, 'g':2, 'm':3, 'd':4, 'p':5, 'w':6, 'l':7})


dataset['class'].unique()
dataset['class'] = dataset['class'].map({'p':0, 'e':1})

print(dataset.corr(method='pearson'))


import seaborn as sns
sns.heatmap(dataset.corr(),annot=True, fmt="d",cmap="YlGnBu")

X = dataset.iloc[:,1:]
Y = dataset.iloc[:,0]

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

dataset.drop(['cap-shape','bruises','odor','gill-spacing','gill-size','gill-color','habitat','population'
             ,'veil-color','stalk-surface-below-ring','stalk-surface-above-ring'],axis = 1,inplace = True)

X = dataset.iloc[:,1:]
Y = dataset.iloc[:,0]


from sklearn.preprocessing import StandardScaler
sclaer = StandardScaler().fit(X)
X = sclaer.transform(X)

num_folds = 10
seed = 7
scoring = 'accuracy'
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

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


classifier = RandomForestClassifier()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_validation)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_validation, y_pred)
