import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("simulation_500.csv")
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

def predict(X_train,y_train,X_test,k):
    y_pred=[]
    for pt in X_test:
        distance=[]
        for i in range(len(X_train)):
            dist=np.sqrt(np.sum((pt-X_train[i])**2))
            distance.append((dist,i))
        distance.sort(key=lambda x:x[0])
        neighbour=[distance[i][1] for i in range(k)]
        votes={}
        for i in neighbour:
            dist=np.sqrt(np.sum((pt-X_train[i])**2))
# small distance gives big weight and division by 0 shouldnt be possible so we add 0.000000001
            weight=1/(dist+0.000000001)
            label=y_train[i]
            votes[label]=votes.get(label,0)+weight
        prediction=max(votes,key=votes.get)
        y_pred.append(prediction)
    return y_pred

def score(y_test,y_pred):
    correct=0
    for i in range(len(y_test)):
        if y_test[i]==y_pred[i]:
            correct+=1
    return correct/len(y_test)

k_values=[1,3,5,7,9]
weighted=[]
sk=[]
for k in k_values:
    y_pred=predict(X_train,y_train,X_test,k)
    weighted.append(score(y_test,y_pred))
    neigh=KNeighborsClassifier(n_neighbors=k,weights='distance')
    neigh.fit(X_train,y_train)
    sk.append(neigh.score(X_test,y_test))

print("Weighted KNN:",weighted)
print("Sklearn Weighted KNN:",sk)
plt.plot(k_values,weighted,marker='o',label='Own Weighted KNN')
plt.plot(k_values,sk,marker='s',label='Sklearn Weighted KNN')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Weighted KNN Comparison")
plt.legend()
plt.show()