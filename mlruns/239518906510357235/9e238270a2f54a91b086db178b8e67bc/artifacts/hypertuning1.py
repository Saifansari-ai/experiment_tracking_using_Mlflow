import mlflow.data
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

data = load_breast_cancer()
x = pd.DataFrame(data.data,columns=data.feature_names)
y = pd.Series(data.target,name='target')

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

rf = RandomForestClassifier(random_state=42)

param_gird = {
    'n_estimators':[10,20,50],
    'max_depth':['None',10,20,30]
}

grid_search = GridSearchCV(estimator=rf,param_grid=param_gird,cv=5,n_jobs=-1,verbose=2)

# hypertuning without mlflow

# grid_search.fit(X_train,y_train)

# best_param = grid_search.best_params_
# best_score = grid_search.best_score_

# print(best_param)
# print(best_score)

# Hypertuning with mlflow

mlflow.set_experiment("breast_cancer_rf_hy")

with mlflow.start_run():
    grid_search.fit(X_train,y_train)

    best_param = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_param)
    mlflow.log_metric('accuracy',best_score)

    train_df = X_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,"training")

    test_df = X_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,"testing")

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(grid_search.best_estimator_,"random forest")

    mlflow.set_tag('auther','Saifansari')

    print(best_param)
    print(best_score)
