import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

print("Loading contextual base...")
df = pd.read_parquet('outputs/contextual_base.parquet')

# STEP 2: CONTEXT BASELINE
print("Calculating Context Baselines (Robust Z-Scores)...")
df['total_request_volume_log'] = np.log1p(df['total_request_volume'].fillna(0))
df['weighted_avg_latency_log'] = np.log1p(df['weighted_avg_latency'].fillna(0))
df['client_request_volume_log'] = np.log1p(df['client_request_volume'].fillna(0))
df['server_request_volume_log'] = np.log1p(df['server_request_volume'].fillna(0))
df['error_request_volume_log'] = np.log1p(df['error_request_volume'].fillna(0))

features = [
    'total_request_volume_log', 'weighted_avg_latency_log', 
    'client_request_volume_log', 'server_request_volume_log', 'error_request_volume_log'
]

def get_mad(x):
    return np.median(np.abs(x - np.median(x)))

print("Grouping by host and endpoint to find historical norms...")
agg_df = df.groupby(['host', 'endpoint'])[features].agg(['median', get_mad])
agg_df.columns = [f"{col[0]}_{'median' if col[1]=='median' else 'mad'}" for col in agg_df.columns]
df = df.merge(agg_df.reset_index(), on=['host', 'endpoint'], how='left')

# STEP 3: CONTEXTUAL ANOMALY SCORE
print("Calculating contextual anomaly scores...")
for f in features:
    df[f'{f}_z'] = np.abs(df[f] - df[f'{f}_median']) / (1.4826 * df[f'{f}_mad'] + 1e-6)

df['context_anomaly_score'] = df[[f'{f}_z' for f in features]].max(axis=1)

# STEP 4: SYSTEM TIMELINE AGGREGATION
print("Aggregating back to system-wide timeline...")
thresholds_z = [5.0]

for t in thresholds_z:
    df[f'is_anom_{t}'] = (df['context_anomaly_score'] > t).astype(int)

system_df = df.groupby('interval_start').agg(
    max_context_anomaly_score=('context_anomaly_score', 'max'),
    num_contexts=('host', 'count'),
    anom_count_5=('is_anom_5.0', 'sum'),
).reset_index()

system_df['anomalous_context_fraction'] = system_df['anom_count_5'] / system_df['num_contexts']

# Add known anomalies for evaluation
df_anomalies = pd.read_csv('anomaly_windows.csv')
start_dt = pd.to_datetime(df_anomalies['anomaly_start'], utc=True)
end_dt = pd.to_datetime(df_anomalies['anomaly_end'], utc=True)
df_anomalies['start_ts'] = (start_dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
df_anomalies['end_ts'] = (end_dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')

system_df['known_anomaly'] = 0
for _, row in df_anomalies.iterrows():
    mask = (system_df['interval_start'] >= row['start_ts']) & (system_df['interval_start'] <= row['end_ts'])
    system_df.loc[mask, 'known_anomaly'] = 1

# Save system_df
system_df.to_parquet('outputs/system_df_experiment_3.parquet', index=False)
print("System dataframe saved.")

# DISTRIBUTION ANALYSIS
print("\n--- DISTRIBUTION ANALYSIS ---")
group_a = system_df[system_df['known_anomaly'] == 1]['anomalous_context_fraction']
group_b = system_df[system_df['known_anomaly'] == 0]['anomalous_context_fraction']

def print_stats(series, name):
    print(f"Stats for {name}:")
    print(f"  Count: {series.count()}")
    print(f"  Minimum: {series.min():.5f}")
    print(f"  Median: {series.median():.5f}")
    print(f"  90th percentile: {series.quantile(0.90):.5f}")
    print(f"  95th percentile: {series.quantile(0.95):.5f}")
    print(f"  99th percentile: {series.quantile(0.99):.5f}")
    print(f"  99.5th percentile: {series.quantile(0.995):.5f}")
    print(f"  99.9th percentile: {series.quantile(0.999):.5f}")
    print(f"  Maximum: {series.max():.5f}")
    print(f"  Mean: {series.mean():.5f}")

print_stats(group_a, "Inside Known Anomaly Windows")
print_stats(group_b, "Outside Known Anomaly Windows (Normal)")

# THRESHOLD EVALUATION
thresholds = [0.005, 0.01, 0.02, 0.05, 0.10, group_b.quantile(0.99)] 
threshold_names = ['0.5%', '1%', '2%', '5%', '10%', f'Normal 99th ({group_b.quantile(0.99):.5f})']

results = []

print("\n--- THRESHOLD EXPERIMENTS ---")
for t_val, t_name in zip(thresholds, threshold_names):
    pred_anom = (system_df['anomalous_context_fraction'] >= t_val).astype(int)
    y_true = system_df['known_anomaly']
    
    prec = precision_score(y_true, pred_anom, zero_division=0)
    rec = recall_score(y_true, pred_anom, zero_division=0)
    f1 = f1_score(y_true, pred_anom, zero_division=0)
    cm = confusion_matrix(y_true, pred_anom)
    
    detected_windows = 0
    observable_windows = 0
    for idx, row in df_anomalies.iterrows():
        w_start = row['start_ts'] - 300
        w_end = row['end_ts'] + 300
        w_mask = (system_df['interval_start'] >= w_start) & (system_df['interval_start'] <= w_end)
        if w_mask.sum() > 0:
            observable_windows += 1
            if pred_anom[w_mask].sum() > 0:
                detected_windows += 1
                
    pred_anom_count = pred_anom.sum()
    
    results.append({
        'Threshold': t_name,
        'Value': t_val,
        'Precision': f"{prec*100:.2f}%",
        'Recall': f"{rec*100:.2f}%",
        'F1': f"{f1*100:.2f}%",
        'Predicted_Intervals': pred_anom_count,
        'Windows_Detected': detected_windows,
        'Observable_Detected': observable_windows,
        'CM': cm.tolist()
    })

for res in results:
    print(f"Threshold: {res['Threshold']} | Prec: {res['Precision']} | Rec: {res['Recall']} | F1: {res['F1']} | Pred: {res['Predicted_Intervals']} | Win: {res['Windows_Detected']}/25 | Obs: {res['Windows_Detected']}/{res['Observable_Detected']}")

with open('outputs/experiment_3_results.json', 'w') as f:
    json.dump(results, f, indent=4)
