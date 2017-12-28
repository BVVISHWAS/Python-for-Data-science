import pandas as pd
import numpy as np
import math

a = pd.read_csv('D:\\ML\\TITANIC\\New\\Submission_Titanic_GBC.csv')
b = pd.read_csv('D:\\ML\\TITANIC\\New\\Submission_Titanic_NN.csv')
c = pd.read_csv('D:\\ML\\TITANIC\\New\\Submission_Titanic_SVM.csv')

Final_output = pd.concat([ a['Survived'],b['Survived'],c['Survived']], axis=1)


output  =Final_output.mode(axis = 1)


csv_output = pd.concat([ a['PassengerId'],output], axis=1)

csv_output.to_csv("D:\\ML\\TITANIC\\New\\Submission_Titanic.csv", index=False)
