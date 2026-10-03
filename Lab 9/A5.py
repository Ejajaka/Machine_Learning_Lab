import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lime.lime_tabular import LimeTabularExplainer

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
    pipe=Pipeline([("scaler",StandardScaler()),("model",SVC(probability=True))])
    return pipe
# Train the pipeline using the training data
def trainPipeline(pipe,Xtrain,ytrain):
    pipe.fit(Xtrain,ytrain)
    return pipe
# Create the LIME explainer using the training data
def createExplainer(Xtrain,ytrain):
    explainer=LimeTabularExplainer(
        Xtrain.values,
        feature_names=Xtrain.columns.tolist(),
        class_names=sorted(ytrain.unique().astype(str)),
        mode="classification",
        random_state=42
    )
    return explainer
# Explain one prediction using LIME
def explainPrediction(explainer,pipe,Xtest,index):
    instance=Xtest.iloc[index].values
    exp=explainer.explain_instance(instance,pipe.predict_proba,num_features=10)
    pred=pipe.predict(Xtest.iloc[[index]])[0]
    return pred,exp

fp="ACTG175_X_train.csv"
df=loadData(fp)
X,y=prepData(df)
Xtrain,Xtest,ytrain,ytest=splitData(X,y)
pipe=createPipeline()
pipe=trainPipeline(pipe,Xtrain,ytrain)
explainer=createExplainer(Xtrain,ytrain)
pred,exp=explainPrediction(explainer,pipe,Xtest,0)
print("Predicted Class:",pred)
print("\nLIME Explanation:")
for feature,weight in exp.as_list():
    print(feature,":",weight)