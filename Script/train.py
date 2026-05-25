import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import yaml
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def model_train():
    config = load_config()
    
    target_var = os.environ.get("TARGET_VARIABLE", "Stock Price")
    model_type = os.environ.get("MODEL_TYPE", "RandomForest")
    visualization_type = os.environ.get("VISUALIZATION_TYPE", "Scatter Plot")
    ticker = os.environ.get("TICKER", "AAPL")
    
    mlflow.set_experiment(f"Financial_Prediction")
    with mlflow.start_run():
        mlflow.set_tag("Target", target_var)
        mlflow.set_tag("Model", model_type)
        mlflow.set_tag("Visualization", visualization_type)
        mlflow.set_tag("Ticker", ticker)
        
        print(f"Loading data from {config['data']['raw_data_path']}...")
        try:
            data = pd.read_csv(config['data']['raw_data_path'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Data not found. Did you run fetch_data.py?")
            
        target_col = config['train']['target_column']
        X = data.drop([target_col], axis=1)
        y = data[target_col]
        
        test_size = config['train']['test_size']
        random_state = config['train']['random_state']
        
        # We don't stratify for regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training {model_type}...")
        
        if model_type == "RandomForest":
            estimator = RandomForestRegressor(random_state=random_state)
            param_grid = config['train']['param_grids'].get('RandomForest', {})
        elif model_type == "GradientBoosting":
            estimator = GradientBoostingRegressor(random_state=random_state)
            param_grid = config['train']['param_grids'].get('GradientBoosting', {})
        elif model_type == "LinearRegression":
            estimator = LinearRegression()
            param_grid = config['train']['param_grids'].get('LinearRegression', {})
        elif model_type == "SVR":
            estimator = SVR()
            param_grid = config['train']['param_grids'].get('SVR', {})
        elif model_type == "KNeighbors":
            estimator = KNeighborsRegressor()
            param_grid = config['train']['param_grids'].get('KNeighbors', {})
        elif model_type == "DecisionTree":
            estimator = DecisionTreeRegressor(random_state=random_state)
            param_grid = config['train']['param_grids'].get('DecisionTree', {})
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        n_samples = len(X_train)
        cv_splits = min(3, max(2, n_samples))
        
        if n_samples < 2:
            print("Dataset too small for Cross-Validation. Training model directly.")
            best_model = estimator
            best_model.fit(X_train, y_train)
            best_params = {}
        else:
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv_splits,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        mlflow.log_params(best_params)
        
        print("Evaluating Model...")
        predictions = best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        mlflow.log_metrics(metrics)
        print(f"Metrics: {metrics}")
        
        # Generate Plot
        reports_dir = config.get('reports', {}).get('dir', 'reports')
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
            
        plt.figure(figsize=(10, 6))
        
        if visualization_type == "Scatter Plot":
            sns.scatterplot(x=y_test, y=predictions, alpha=0.6, color='blue', label='Predictions')
            min_val = min(y_test.min(), predictions.min())
            max_val = max(y_test.max(), predictions.max())
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
            plt.title(f"Actual vs Predicted {target_var} ({ticker}) - {model_type}")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.legend()
            
        elif visualization_type == "Residual Plot":
            sns.residplot(x=predictions, y=y_test, color='purple', lowess=True)
            plt.title(f"Residual Plot for {target_var} ({ticker}) - {model_type}")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals (Actual - Predicted)")
            
        elif visualization_type == "Error Histogram":
            errors = y_test - predictions
            sns.histplot(errors, kde=True, color='orange', bins=30)
            plt.title(f"Prediction Error Distribution ({ticker}) - {model_type}")
            plt.xlabel("Prediction Error")
            plt.ylabel("Frequency")
            
        elif visualization_type == "Line Chart":
            plt.plot(y_test.values, label='Actual Values', color='green', marker='o', alpha=0.7)
            plt.plot(predictions, label='Predicted Values', color='red', marker='x', alpha=0.7)
            plt.title(f"Actual vs Predicted Line Chart ({ticker}) - {model_type}")
            plt.xlabel("Test Samples (Chronological Order)")
            plt.ylabel(f"{target_var}")
            plt.legend()
            
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(reports_dir, "prediction_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(plot_path)
        print(f"Prediction plot saved to {plot_path}")
        
        formatted_params = json.dumps(best_params, indent=2)
        metrics_md = f"""## 📊 Financial Prediction Metrics ({target_var})
### 📈 Ticker: {ticker} | 🤖 Model: {model_type}

| Metric | Score |
|---|---|
| **R² Score** | `{r2:.4f}` |
| **RMSE** | `{rmse:.4f}` |
| **MAE** | `{mae:.4f}` |
| **MSE** | `{mse:.4f}` |

### 🛠️ Best Hyperparameters
```json
{formatted_params}
```
"""
        with open("metrics.md", "w", encoding="utf-8") as f:
            f.write(metrics_md)
        print("Exported metrics to metrics.md")
        
        mlflow.sklearn.log_model(best_model, "financial_model")
        
        model_dir = config['model']['model_dir']
        model_path = config['model']['model_path']
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    model_train()