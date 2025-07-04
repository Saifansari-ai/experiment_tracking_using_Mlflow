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

grid_search.fit(X_train,y_train)

best_param = grid_search.best_params_
best_score = grid_search.best_score_

print(best_param)
print(best_score)