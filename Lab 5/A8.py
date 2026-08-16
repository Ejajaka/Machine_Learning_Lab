import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("simulation_500.csv")
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

# to measure similarity between target and data points use eucliedian
def eucl(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
# Identifying neighbors
def knn(data,pt,k):
    distance=[]
    for i in range(len(data)):
        dist=eucl(pt,data[i])
        distance.append((dist,i))
    # Sorting
    distance.sort(key=lambda x:x[0])
    neighbour=[distance[i][1] for i in range(k)]
    return neighbour

# prediction
def predict(X_train,y_train,X_test,k):
    y_pred=[]
    for pt in X_test:
        neighbour=knn(X_train,pt,k)
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

k_values=[1,3,5,7,9]
own=[]
sk=[]
for k in k_values:
    y_pred=predict(X_train,y_train,X_test,k)
    own.append(score(y_test,y_pred))
    neigh=KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train,y_train)
    sk.append(neigh.score(X_test,y_test))
print("Own:",own)
print("Sk:",sk)

plt.plot(k_values,own,marker='o',label='Own')
plt.plot(k_values,sk,marker='s',label='Sk')
plt.xlabel("k")
plt.ylabel("Acc")
plt.title("Acc vs k")
plt.legend()
plt.show()