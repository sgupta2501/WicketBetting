import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_excel('step3_PlayerRanks.xlsx')

a=input("Enter the no of batsmen you want in indian team (between 5 to 7)\n")
a=int(a)

#MatchNo	asBataman	asBowler	asFielder	asAllRounder	Dream11Score

x= dataset.iloc[0:20,[1,2,3,4,5,6]].values
print("training data")
print(x)

y= dataset.iloc[0:20,0].values
print(y)


#fitting simple linear regression to the training set
#feature scalling
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x=sc.fit_transform(x)

#predicting the test set resuts

x1= dataset.iloc[21:25,[1,2,3,4,5,6]].values
print("test data")
print(x1)

from sklearn.svm import SVC
df=SVC(kernel='rbf',random_state=0)
df.fit(x,y)
y_pred=df.predict(x1)

print(y_pred)

s= []
for i in y_pred:
    if i not in s:
        s.append(i)
batsman=[]

if (a>len(s)):
    a=len(s)
for i in range(0,a):
    batsman.append(s[i])
print(batsman)

