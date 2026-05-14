import pandas as pd
from evidently.report import Report
from evidently.presets import DataDriftPreset, TargetDriftPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric

def check_for_drift():
    # 1. Load your 'Gold Standard' training data
    reference_data = pd.read_csv("Data/Iris.csv").drop(['Id'], axis=1)
    
    # 2. Load the 'Real World' data (Simulate this with a slightly modified CSV)
    # In a real system, you'd pull this from your production_logs.log
    current_data = pd.read_csv("Data/Iris.csv").drop(['Id'], axis=1)
    current_data['SepalLengthCm'] = current_data['SepalLengthCm'] * 1.5 # Simulating drift
    
    # 3. Generate the Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # 4. Save the report as an HTML dashboard
    drift_report.save_html("reports/drift_report.html")
    print("Drift report generated in /reports folder.")

if __name__ == "__main__":
    check_for_drift()