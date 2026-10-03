import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def loadData(fp):
    df=pd.read_csv(fp)
    return df
# Prepare the input and target data
def prepData(df):
    X=df.drop(columns=["arms","Unnamed: 0"])
    y=df["arms"]
    return X,y
# Split the data into training and testing sets
def splitData(X,y):
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return Xtrain,Xtest,ytrain,ytest
# Create the pipeline for scaling the data and training the SVM
def createPipeline():
    pipe=Pipeline([("scaler",StandardScaler()),("model",SVC())])
    return pipe
# Train the pipeline using the training data
def trainPipeline(pipe,Xtrain,ytrain):
    pipe.fit(Xtrain,ytrain)
    return pipe
# Evaluate the model using the test data
def evaluatePipeline(pipe,Xtest,ytest):
    yp=pipe.predict(Xtest)
    acc=accuracy_score(ytest,yp)
    pre=precision_score(ytest,yp,average="weighted",zero_division=0)
    rec=recall_score(ytest,yp,average="weighted",zero_division=0)
    f1=f1_score(ytest,yp,average="weighted",zero_division=0)
    return acc,pre,rec,f1

fp="ACTG175_X_train.csv"
df=loadData(fp)
X,y=prepData(df)
Xtrain,Xtest,ytrain,ytest=splitData(X,y)
pipe=createPipeline()
pipe=trainPipeline(pipe,Xtrain,ytrain)
acc,pre,rec,f1=evaluatePipeline(pipe,Xtest,ytest)
print("Pipeline: StandardScaler + SVM")
print("Accuracy:",acc)
print("Precision:",pre)
print("Recall:",rec)
print("F1-Score:",f1)