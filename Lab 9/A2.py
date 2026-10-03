import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def loadData(file):
    df=pd.read_csv(file)
    return df
# Prepare the data by separating the input features and target
def prepData(df):
    X=df.drop(columns=["arms","Unnamed: 0"])
    y=df["arms"]
    return X,y
# Split the data into training and testing sets
def splitData(X,y):
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return Xtrain,Xtest,ytrain,ytest
# Scale the features so that they are on a similar scale
def scaleData(Xtr,Xte):
    scaler=StandardScaler()
    Xtr=scaler.fit_transform(Xtr)
    Xte=scaler.transform(Xte)
    return Xtr,Xte
# Tune the Perceptron to find suitable hyperparameters
def tunePrc(Xtr,ytr):
    model=Perceptron(random_state=42)
    prm={"penalty":[None,"l2","l1","elasticnet"],"alpha":np.logspace(-6,-2,10),"max_iter":[500,1000,1500,2000],"tol":[1e-2,1e-3,1e-4],"eta0":[0.01,0.1,1.0,10.0]}
    rs=RandomizedSearchCV(model,prm,n_iter=20,cv=5,scoring="accuracy",random_state=42,n_jobs=-1)
    rs.fit(Xtr,ytr)
    return rs
# Tune the MLP by trying different hyperparameter combinations
def tuneMlp(Xtr,ytr):
    model=MLPClassifier(random_state=42)
    prm={"hidden_layer_sizes":[(50,),(100,),(50,50),(100,50),(100,100)],"activation":["identity","tanh","relu"],"solver":["adam","sgd"],"alpha":np.logspace(-5,-2,10),"learning_rate_init":[0.0001,0.001,0.01],"batch_size":[32,64,128],"max_iter":[300,500,700]}
    rs=RandomizedSearchCV(model,prm,n_iter=20,cv=5,scoring="accuracy",random_state=42,n_jobs=-1)
    rs.fit(Xtr,ytr)
    return rs
# Evaluate the model using the test data
def evalModel(m,Xte,yte):
    yp=m.predict(Xte)
    acc=accuracy_score(yte,yp)
    return acc

fp="ACTG175_X_train.csv"
df=loadData(fp)
X,y=prepData(df)
Xtr,Xte,ytr,yte=splitData(X,y)
Xtr,Xte=scaleData(Xtr,Xte)
prs=tunePrc(Xtr,ytr)
mls=tuneMlp(Xtr,ytr)
prcAcc=evalModel(prs.best_estimator_,Xte,yte)
mlpAcc=evalModel(mls.best_estimator_,Xte,yte)
print("Perceptron Best Parameters:")
print(prs.best_params_)
print("\nPerceptron CV Accuracy:")
print(prs.best_score_)
print("\nPerceptron Test Accuracy:")
print(prcAcc)
print("\nMLP Best Parameters:")
print(mls.best_params_)
print("\nMLP CV Accuracy:")
print(mls.best_score_)
print("\nMLP Test Accuracy:")
print(mlpAcc)