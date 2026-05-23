import pandas as pd
import yaml
import os
import subprocess
import sys
import requests
from evidently.report import Report
from evidently.presets import DataDriftPreset

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def check_for_drift_and_retrain():
    config = load_config()
    ref_data_path = config['data']['raw_data_path']
    live_data_path = "Data/live_data.csv"
    
    if not os.path.exists(live_data_path):
        print(f"No user contributions collected yet at {live_data_path}. Skipping drift check.")
        return
        
    print(f"Loading original training data from {ref_data_path}...")
    reference_data = pd.read_csv(ref_data_path)
    
    print(f"Loading contributed live data from {live_data_path}...")
    current_data = pd.read_csv(live_data_path)
    
    # We only want to trigger retraining if we have enough new data to make it statistically meaningful
    min_samples = 10
    if len(current_data) < min_samples:
        print(f"Only {len(current_data)} new contributions. Waiting for at least {min_samples} before checking for drift.")
        return
        
    # Drop target column for drift detection
    target_col = config['train']['target_column']
    ref_features = reference_data.drop([target_col], axis=1)
    curr_features = current_data.drop([target_col], axis=1)
    
    print("Generating drift report to compare contributions against original data...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref_features, current_data=curr_features)
    
    reports_dir = config['reports']['dir']
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        
    drift_report.save_html(config['reports']['drift_report_path'])
    print(f"Drift HTML dashboard saved to {config['reports']['drift_report_path']}")
    
    # Parse the drift result programmatically
    drift_result = drift_report.as_dict()
    dataset_drift = drift_result['metrics'][0]['result']['dataset_drift']
    
    if dataset_drift:
        print("🚨 SIGNIFICANT DATA DRIFT DETECTED! The contributed data has introduced new patterns.")
        print("Triggering Continuous Training pipeline...")
        
        # 1. Merge live data into reference data
        merged_data = pd.concat([reference_data, current_data], ignore_index=True)
        merged_data.to_csv(ref_data_path, index=False)
        print(f"✅ Appended {len(current_data)} new contributed rows to the main dataset.")
        
        # 2. Clear live data so we don't retrain on it again
        os.remove(live_data_path)
        print("✅ Cleared live_data.csv queue.")
        
        # 3. Trigger retraining via subprocess
        print("🚀 Launching Script/train.py in the background to retrain the AI...")
        subprocess.run([sys.executable, "Script/train.py"], check=True)
        
        # 4. Tell the running API to hot-swap the new model
        print("🔄 Hot-swapping the new model into the live API...")
        try:
            response = requests.post("http://127.0.0.1:8000/ReloadModel")
            if response.status_code == 200:
                print("✅ Continuous Training complete! The new model is now active with zero downtime.")
            else:
                print("⚠️ Retraining succeeded, but API hot-swap failed. Please restart the server manually.")
        except Exception as e:
            print(f"⚠️ Could not reach the API to hot-swap. Is the server running? ({e})")
        
    else:
        print("✅ Contributions match existing patterns. No significant drift detected. Model is still healthy.")

if __name__ == "__main__":
    check_for_drift_and_retrain()