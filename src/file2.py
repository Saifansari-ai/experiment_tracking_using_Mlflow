import mlflow
import mlflow.sklearn
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import joblib
import os

dagshub.init(repo_owner='Saifansari-ai', repo_name='experiment_tracking_using_Mlflow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Saifansari-ai/experiment_tracking_using_Mlflow.mlflow")

# load dataset
wine = load_wine()
x = wine.data
y = wine.target

# splitting data for the training
X_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=42)

# setting up parameter
max_depth = 5
n_estimators = 5

mlflow.set_experiment("mlflow-exp_2")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)
    print(accuracy)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

    # Creating confusion metrix plot
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion metrix')

    # save plot
    plt.savefig("Confusion_metrix_2.png")

    # log artifact using mlflow
    mlflow.log_artifact("Confusion_metrix_2.png")
    mlflow.log_artifact(__file__)

    mlflow.set_tags({'Auther':'Saif','Project':'Wine classification'})

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/random_forest_2.pkl")

    mlflow.log_artifact("models/random_forest_2.pkl", artifact_path="model")

