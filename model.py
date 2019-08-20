# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:16:20 2019

@author: raahul46
"""
####DEPENDENCIES####
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from preprocess import preprocessing

####PREPROCESSING OF DATASET####
x_train,y_train,x_test,y_test = preprocessing()

####MODEL TRAINING####
print("...training model...")        
classifier1=RandomForestClassifier(n_estimators=300,criterion="entropy",)
classifier1.fit(x_train,y_train)

####PREDICTION####
y_pred=classifier1.predict(x_test)
    
#SAVING THE MODEL(Pickle File)
filename = 'RF_model_final1.sav'
pickle.dump(classifier1, open(filename, 'wb'))

#Train-Validation-Test
accuracies=cross_val_score(estimator=classifier1,X=x_train,y=y_train,cv=10) 

#INSIGHTS & INFERENCES
print("Accuracy of the model: ",metrics.accuracy_score(y_test, y_pred)*100)
print("Mean Accuracy: ",accuracies.mean()*100)
print("Standard Deviation: ",accuracies.std()*100)


#Confusion matrix
print(metrics.confusion_matrix(y_test, y_pred))



