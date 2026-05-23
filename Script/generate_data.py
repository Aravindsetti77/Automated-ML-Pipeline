import pandas as pd
import numpy as np
import os

def generate_fraud_data(num_samples=100000):
    np.random.seed(42)
    
    # Generate realistic continuous features
    distance_from_home = np.abs(np.random.normal(50, 200, num_samples))
    distance_from_last_transaction = np.abs(np.random.normal(10, 50, num_samples))
    ratio_to_median_purchase_price = np.abs(np.random.normal(1, 5, num_samples))
    
    # Generate categorical features (0 or 1)
    repeat_retailer = np.random.choice([0, 1], num_samples, p=[0.12, 0.88])
    used_chip = np.random.choice([0, 1], num_samples, p=[0.65, 0.35])
    used_pin_number = np.random.choice([0, 1], num_samples, p=[0.90, 0.10])
    online_order = np.random.choice([0, 1], num_samples, p=[0.35, 0.65])
    
    # Apply logical rules to inject "Fraud" (target variable)
    fraud = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        prob = 0.01 # Base probability of fraud is 1%
        
        if online_order[i] == 1:
            prob += 0.05
            
        if used_pin_number[i] == 0:
            prob += 0.05
            
        if ratio_to_median_purchase_price[i] > 10:
            prob += 0.20
            
        if distance_from_home[i] > 200:
            prob += 0.10
            
        if used_chip[i] == 0 and online_order[i] == 0:
            prob += 0.10
            
        # Determine if this transaction is fraudulent based on accumulated probability
        if np.random.random() < prob:
            fraud[i] = 1
            
    df = pd.DataFrame({
        'distance_from_home': distance_from_home,
        'distance_from_last_transaction': distance_from_last_transaction,
        'ratio_to_median_purchase_price': ratio_to_median_purchase_price,
        'repeat_retailer': repeat_retailer,
        'used_chip': used_chip,
        'used_pin_number': used_pin_number,
        'online_order': online_order,
        'fraud': fraud
    })
    
    if not os.path.exists('Data'):
        os.makedirs('Data')
        
    output_path = "Data/synthetic_fraud_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} transactions.")
    print(f"Total Fraud cases: {fraud.sum()}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    generate_fraud_data()
