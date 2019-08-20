# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:54:48 2019

@author: raahul46
"""
####DEPENDENCIES####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

####PRE PROCESSING FUNCTION####
def preprocessing():
    
    ####READING DATASETS####
    result = pd.read_csv("gender_submission.csv")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    ####PASSENGER ID####
    train_id = train[:0]
    test_id = test[:0]
    
    ####DATASET MERGE####
    test = test.merge(result, on = "PassengerId")
    train = train.drop(["PassengerId"],axis = 1)
    test = test.drop(["PassengerId"],axis = 1)


    train = train.drop(["Name","Ticket","Cabin"],axis = 1)

    ####MISSING VALUES####
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        
    train['Age'].fillna(train['Age'].mode()[0],inplace=True)
    print(train['Age'].value_counts())
        
    train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
    print(train["Embarked"].value_counts())    
    
    print(train.columns.hasnans)

    train_f = train.copy()
    
    ####ENCODING CATEGORICAL DATA####
    lb=LabelBinarizer()
    list2=list(train_f.columns.values)

            
    lb_results = lb.fit_transform(train_f["Embarked"])
    lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)
    print(lb_results_df.head())
    train_f = pd.concat([train_f, lb_results_df], axis=1)
            
    lb_results = (lb.fit_transform(train_f["Sex"]))
    z = np.zeros((891,1), dtype= "int32")
    lb_results = np.append(lb_results,z, axis = 1)
    lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)
    print(lb_results_df.head())
    train_f = pd.concat([train_f, lb_results_df], axis=1)       

    list2=list(train_f.columns.values)

    print("Removing orginal cateogorical variables...")
    for i in [2,7,12]:
                 j=list2[i]
                 print(j)
                 train_f = train_f.drop([j],axis=1)
    list2=list(train_f.columns.values)

    ####TRAINING SET####
    y = train_f.iloc[:,0].values
    x = train_f.iloc[:,1:].values 
    list4=[]
    for i in range(9):
        list4.append(np.amax(x[:,i]))
    x_train = np.copy(x)
    y_train = np.copy(y)

    ####CO RELATION INSIGHTS####
    list2.remove("Survived")
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(x,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=list2)
    feat_importances.nlargest(20).plot(kind='barh')
    plt.show()
        
        
####################### TEST DATASET #######################################


    test = test.drop(["Name","Ticket","Cabin"],axis = 1)
    
    ####MISSING VALUES####
    tota1l = test.isnull().sum().sort_values(ascending=False)
    percent1 = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
    missing_data1 = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        
    test['Age'].fillna(test['Age'].mode()[0],inplace=True)
    print(test['Age'].value_counts())
        
    test['Embarked'].fillna(test['Embarked'].mode()[0],inplace=True)
    print(test['Embarked'].value_counts())
    
    test['Fare'].fillna(test['Fare'].mode()[0],inplace=True)
    print(test['Embarked'].value_counts())    
    
    print(test.columns.hasnans)
    test_f = test.copy()
    
    ####ENCODING CATEGORICAL DATA####
    lb=LabelBinarizer()
    list21=list(test_f.columns.values)

            
    lb_results1 = lb.fit_transform(test_f["Embarked"])
    lb_results_df1 = pd.DataFrame(lb_results1, columns=lb.classes_)
    print(lb_results_df1.head())
    test_f = pd.concat([test_f, lb_results_df1], axis=1)
            
    lb_results1 = (lb.fit_transform(test_f["Sex"]))
    z1 = np.zeros((418,1), dtype= "int32")
    lb_results1 = np.append(lb_results1,z1, axis = 1)
    lb_results_df1 = pd.DataFrame(lb_results1, columns=lb.classes_)
    print(lb_results_df1.head())
    test_f = pd.concat([test_f, lb_results_df1], axis=1)       


    list2=list(test_f.columns.values)

    print("Removing orginal cateogorical variables...")
    for i in [1,6,12]:
             j=list2[i]
             print(j)
             test_f = test_f.drop([j],axis=1)
    cols_to_move = ['Survived']
    new_cols = np.hstack((test_f.columns.difference(cols_to_move), cols_to_move))
    test_f = test_f.reindex(columns=new_cols)

    print(test_f.columns.hasnans)
    
    ####TEST SET####
    y1 = test_f.iloc[:,-1].values
    x1 = test_f.iloc[:,0:9].values 
    list41=[]
    for i in range(9):
        list41.append(np.amax(x1[:,i]))
    x_test = np.copy(x1)
    y_test = np.copy(y1)

    return x_train,y_train,x_test,y_test


