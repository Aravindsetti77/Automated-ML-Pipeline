Orchestration: Python-based trigger scripts for automated training cycles.

Experiment Tracking: MLflow for logging hyperparameters (GridSearchCV), metrics, and .joblib artifacts.

Model Serving: FastAPI providing a high-performance REST endpoint for real-time inference.

Data Versioning: DVC (Data Version Control) to snapshot the Kaggle Iris dataset.

Model Monitoring: Evidently AI for detecting Data Drift and Target Drift in production logs.

Ingestion & Versioning: Raw Kaggle CSVs are tracked with DVC to ensure every model version is linked to a specific data state.

Automated Training: train.py executes GridSearchCV to optimize hyperparameters, logging the "Best Parameters" automatically to the MLflow dashboard.

Deployment: The "Champion" model is serialized and picked up by a FastAPI server, enabling real-time predictions via Swagger UI.

Logging: Every prediction is recorded in production_logs.log to create an audit trail of live performance.

Monitoring: monitor.py compares live logs against the training baseline. If the statistical distribution shifts (Drift), the system flags the model for a maintenance cycle.

Some Samples for the accuracy of the pipeline:
<img width="1857" height="872" alt="image" src="https://github.com/user-attachments/assets/1e454e1a-e566-40f9-a31b-57286f82326a" />
