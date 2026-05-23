import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import joblib
import os
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def model_train():
    config = load_config()
    
    mlflow.set_experiment("credit_card_fraud_detection")
    with mlflow.start_run():
        print(f"Loading data from {config['data']['raw_data_path']}...")
        data = pd.read_csv(config['data']['raw_data_path'])
        
        target_col = config['train']['target_column']
        X = data.drop([target_col], axis=1)
        y = data[target_col]
        
        test_size = config['train']['test_size']
        random_state = config['train']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        param_grid = config['train']['param_grid']
        
        print("Starting Grid Search...")
        rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,  # Reduced cv for faster training on large dataset
            scoring="f1",
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)
        
        print("Evaluating Model...")
        predictions = best_model.predict(X_test)
        predict_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "f1_score": f1_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, predict_proba)
        }
        mlflow.log_metrics(metrics)
        print(f"Metrics: {metrics}")
        
        # Write metrics to a markdown file for GitHub Actions Summary
        metrics_md = f"""## 📊 Model Training Metrics

| Metric | Score |
|---|---|
| **F1 Score** | `{metrics['f1_score']:.4f}` |
| **Precision** | `{metrics['precision']:.4f}` |
| **Recall** | `{metrics['recall']:.4f}` |
| **ROC AUC** | `{metrics['roc_auc']:.4f}` |
"""
        with open("metrics.md", "w", encoding="utf-8") as f:
            f.write(metrics_md)
        print("Exported metrics to metrics.md")
        
        mlflow.sklearn.log_model(best_model, "fraud_rf_model")
        
        model_dir = config['model']['model_dir']
        model_path = config['model']['model_path']
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    model_train()