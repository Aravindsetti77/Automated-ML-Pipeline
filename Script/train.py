import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from sklearn.preprocessing import LabelEncoder
def model_train():
    mlflow.set_experiment("iris_model")
    with mlflow.start_run():
        data=pd.read_csv("Data/Raw_data.csv")
        X=data.drop(["Id","Species"],axis=1)
        le = LabelEncoder()
        y = le.fit_transform(data['Species'])
        joblib.dump(le.classes_, "models/classes.joblib")
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=41)
        param_grid={
            'n_estimators': list(range(50, 350, 50)),
            'max_depth':list(range(5,35,5))
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search=GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train,y_train)
        best_depth=grid_search.best_params_.get("max_depth",None)
        best_estimators=grid_search.best_params_.get("n_estimators",None)
        mlflow.log_params({
            "n_estimators": best_estimators,
            "max_depth" : best_depth
        })
        rf.set_params(**grid_search.best_params_)
        rf.fit(X_train,y_train)
        predictions = rf.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "iris_rf_model")
        joblib.dump(rf, "model/iris_model.joblib")
if __name__ == "__main__":
    if not os.path.exists('model'):
        os.makedirs('model')
    model_train()