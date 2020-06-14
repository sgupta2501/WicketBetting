import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import accuracy_score
import pickle


def train_data_kn(x,y,filename):
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

    #feature scalling
    #sc=StandardScaler()
    
    #fitting simple linear regression to the training set    
    #x_train=sc.fit_transform(x_train)
    #x_test=sc.fit_transform(x_test)

    #k neighbors
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(x_train, y_train)
    ypred = clf.predict(x_test)
    # Calculate the absolute errors
    errors = abs(ypred - y_test)# Print out the mean absolute error (mae)
    print('k neighbors  Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #accuracy
    acc = accuracy_score(y_true=y_test, y_pred=ypred)
    print('KNeighborsClassifier   Acc: {:.4f}'.format(acc))
    #confusion matrix
    print('KNeighborsClassifier   Confusion matrix:')
    print(confusion_matrix(y_test, ypred))

    pickle.dump(clf, open(filename, 'wb'))
    # some time later...
    # load the model from disk
    print("loading model...")
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)    
    
def train_data_naivebayes(x,y,filename):
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

    #feature scalling
    #sc=StandardScaler()
    
    #fitting simple linear regression to the training set    
    #x_train=sc.fit_transform(x_train)
    #x_test=sc.fit_transform(x_test)
    
    #naive bayes
    from sklearn.naive_bayes import GaussianNB
    #create an object of the type GaussianNB
    gnb = GaussianNB()
    #train the algorithm on training data and predict using the testing data
    ypred = gnb.fit(x_train, y_train).predict(x_test)
    # Calculate the absolute errors
    errors = abs(ypred - y_test)# Print out the mean absolute error (mae)
    print('Naive-Bayes  Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #print the accuracy score of the model
    print("Naive-Bayes accuracy : ",accuracy_score(y_test, ypred, normalize = True))    
    #confusion matrix
    print('Naive-Bayes   Confusion matrix:')
    print(confusion_matrix(y_test, ypred))

    pickle.dump(gnb, open(filename, 'wb'))
    # some time later...
    # load the model from disk
    print("loading model...")
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)    
    
def train_data_percep(x,y,filename):
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

    #feature scalling
    #sc=StandardScaler()
    
    #fitting simple linear regression to the training set    
    #x_train=sc.fit_transform(x_train)
    #x_test=sc.fit_transform(x_test)
    
    #perceptron
    from sklearn.linear_model import Perceptron   
    p = Perceptron(random_state=42, max_iter=10000, tol=0.001)
    p.fit(x_train, y_train)
    ypred = p.predict(x_test)
    # Calculate the absolute errors
    errors = abs(ypred - y_test)# Print out the mean absolute error (mae)
    print('perceptron  Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #accuracy
    acc = accuracy_score(y_true=y_test, y_pred=ypred)
    print('perceptron   Acc: {:.4f}'.format(acc))
    #confusion matrix
    print('perceptron   Confusion matrix:')
    print(confusion_matrix(y_test, ypred))

    pickle.dump(p, open(filename, 'wb'))
    # some time later...
    # load the model from disk
    print("loading model...")
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)    
    
def train_data_svm(x,y,filename):
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

    #feature scalling
    #sc=StandardScaler()
    
    #fitting simple linear regression to the training set    
    #x_train=sc.fit_transform(x_train)
    #x_test=sc.fit_transform(x_test)

    #SVM
    from sklearn.svm import SVC   
    svm=SVC(kernel='rbf',random_state=0,probability=True)
    svm.fit(x_train, y_train)
    ypred=svm.predict(x_test)
    # Calculate the absolute errors
    errors = abs(ypred - y_test)# Print out the mean absolute error (mae)
    print('SVM   Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #accuracy
    acc = accuracy_score(y_true=y_test, y_pred=ypred)
    print('SVM   Acc: {:.4f}'.format(acc))
    #confusion matrix
    print('SVM   Confusion matrix:')
    print(confusion_matrix(y_test, ypred))
    

    pickle.dump(svm, open(filename, 'wb'))
    # some time later...
    # load the model from disk
    print("loading model...")
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)    
    
