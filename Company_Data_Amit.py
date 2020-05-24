# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:37:51 2020

@author: amit kumar
"""


import pandas as pd
from sklearn.tree import DecisionTreeClassifier #importing decision tree classifier
from sklearn.model_selection import train_test_split #importing train_test_split function
from sklearn.metrics import accuracy_score#importing metrics for accuracy calculation (confusion matrix)
from sklearn.ensemble import BaggingClassifier#bagging combines the results of multipls models to get a generalized result. 
from sklearn.ensemble import AdaBoostClassifier #boosting method attempts to correct the errors of previous models.
#from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
#from IPython.display import Image
#from pydot import graph_from_dot_data
from sklearn.metrics import classification_report, confusion_matrix
#reading the dataset
company=pd.read_csv("D:/Data_Science/Data_Sci_Assignment/Decision Trees/Decision_Tree-master_S/Company_data/Company_data.csv")
company.head()
#viewing the types
company.dtypes
#-------------------converting from categorical data---------------------------
company['High'] = company.Sales.map(lambda x: 1 if x>8 else 0)
company['ShelveLoc']=company['ShelveLoc'].astype('category')
company['Urban']=company['Urban'].astype('category')
company['US']=company['US'].astype('category')
company.dtypes
company.head()

#label encoding to convert categorical values into numeric.
company['ShelveLoc']=company['ShelveLoc'].cat.codes
company['Urban']=company['Urban'].cat.codes
company['US']=company['US'].cat.codes
company.tail()
#------------------------ setting feature and target variables -------------------------------------------------------------#
feature_cols=['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']
#x = company.drop(['Sales', 'High'], axis = 1)
x = company[feature_cols]
y = company.High
print(x)
print(y)
#------------------------ splitting into train and test data -------------------------------------------------------------#
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2,random_state=0)
#-------------------------building decision tree model-----------------------------# 
dcmodel =  BaggingClassifier(DecisionTreeClassifier(max_depth = 6), random_state=0) #decision tree classifier object
dcmodel =  AdaBoostClassifier(DecisionTreeClassifier(max_depth = 6), random_state=0) #decision tree classifier object

dcmodel = dcmodel.fit(x_train,y_train) #train decision tree
y_predict = dcmodel.predict(x_test)
#-----------Finding the accuracy------------------------------------#
print("Accuracy : ", accuracy_score(y_test,y_predict)*100 )
#visualizing the decision tree----------------------------------------------#
#dot_data = StringIO()
#export_graphviz(dcmodel,out_file=dot_data,feature_names=feature_cols)

#(graph, )= graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png())
#-----------------------------------------------------------------------------#
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))
