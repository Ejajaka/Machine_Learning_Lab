import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def fit(X_train,y_train):
    return X_train,y_train

def predict(X_train,y_train,X_test):
    y_pred=[]
    for pt in X_test:
        distance=[]
        for i in range(len(X_train)):
            dist=np.sqrt(np.sum((pt-X_train[i])**2))
            distance.append((dist,i))
        distance.sort(key=lambda x:x[0])
        neighbour=[distance[i][1] for i in range(3)]
        labels=[y_train[i] for i in neighbour]
        prediction=max(set(labels),key=labels.count)
        y_pred.append(prediction)
    return y_pred

def score(y_test,y_pred):
    correct=0
    for i in range(len(y_test)):
        if y_test[i]==y_pred[i]:
            correct+=1
    return correct/len(y_test)
data=pd.read_csv("simulation_500.csv")
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
X_train,y_train=fit(X_train,y_train)
y_pred=predict(X_train,y_train,X_test)
print("Predictions:",y_pred)
print("Score:",score(y_test,y_pred))