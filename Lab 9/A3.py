import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
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
# Create the classifiers that need to be compared
def createModels():
    models={"SVM":SVC(),"Decision Tree":DecisionTreeClassifier(random_state=42),"Random Forest":RandomForestClassifier(n_estimators=100,random_state=42),"CatBoost":CatBoostClassifier(iterations=100,verbose=False,random_state=42),"AdaBoost":AdaBoostClassifier(n_estimators=100,random_state=42),"XGBoost":XGBClassifier(n_estimators=100,eval_metric="mlogloss",random_state=42),"Naive Bayes":GaussianNB()}
    return models
# Train all the classifiers on the training data
def trainModels(models,Xtrain,ytrain):
    for name,model in models.items():
        model.fit(Xtrain,ytrain)
    return models
# Calculate the performance metrics for each model
def getMetrics(model,Xtest,ytest):
    yp=model.predict(Xtest)
    acc=accuracy_score(ytest,yp)
    pre=precision_score(ytest,yp,average="weighted",zero_division=0)
    rec=recall_score(ytest,yp,average="weighted",zero_division=0)
    f1=f1_score(ytest,yp,average="weighted",zero_division=0)
    return acc,pre,rec,f1
# Compare the performance of all the classifiers
def compareModels(models,Xtest,ytest):
    results=[]
    for name,model in models.items():
        acc,pre,rec,f1=getMetrics(model,Xtest,ytest)
        results.append([name,acc,pre,rec,f1])
    res=pd.DataFrame(results,columns=["Classifier","Accuracy","Precision","Recall","F1-Score"])
    return res

fp="ACTG175_X_train.csv"
df=loadData(fp)
X,y=prepData(df)
Xtrain,Xtest,ytrain,ytest=splitData(X,y)
scaler=StandardScaler()
XtrainScaled=scaler.fit_transform(Xtrain)
XtestScaled=scaler.transform(Xtest)
models=createModels()
models["SVM"].fit(XtrainScaled,ytrain)
for name,model in models.items():
    if name!="SVM":
        model.fit(Xtrain,ytrain)
res=compareModels(models,XtestScaled,ytest)
print("Classifier Performance Comparison:")
print(res.to_string(index=False))