import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

start_time = time.time()
os.makedirs("outputs", exist_ok=True)

parquet_path = "unpivoted_data.parquet"
csv_path = "anomaly_windows.csv"

print("Calculating system-wide features via DuckDB...")
con = duckdb.connect()

query = f"""
SELECT 
    interval_start,
    SUM(CASE WHEN aggregated_stats_name = 'count' THEN aggregated_stats_value ELSE 0 END) AS total_request_volume,
    SUM(CASE WHEN kind = 'CLIENT' AND aggregated_stats_name = 'count' THEN aggregated_stats_value ELSE 0 END) AS client_request_volume,
    SUM(CASE WHEN kind = 'SERVER' AND aggregated_stats_name = 'count' THEN aggregated_stats_value ELSE 0 END) AS server_request_volume,
    COUNT(DISTINCT host) AS active_hosts,
    COUNT(DISTINCT endpoint) AS active_endpoints,
    AVG(CASE WHEN aggregated_stats_name = 'avg' THEN aggregated_stats_value END) AS mean_recorded_avg_latency,
    SUM(CASE WHEN statusCode >= 500 AND aggregated_stats_name = 'count' THEN aggregated_stats_value ELSE 0 END) AS error_request_volume
FROM '{parquet_path}'
GROUP BY interval_start
ORDER BY interval_start
"""

df_features = con.execute(query).df()
print("DuckDB processing finished.")

print("Saving feature tables...")
df_features.to_parquet("outputs/system_features.parquet", index=False)
df_features.head(100).to_csv("outputs/system_features_preview.csv", index=False)

# TASK 2: Time handling
min_ts = df_features['interval_start'].min()
max_ts = df_features['interval_start'].max()
expected_intervals = int((max_ts - min_ts) / 300) + 1
actual_intervals = len(df_features)
missing_intervals = expected_intervals - actual_intervals
print(f"Intervals - Actual: {actual_intervals}, Expected: {expected_intervals}, Missing: {missing_intervals}")

# TASK 3: Ground-truth labels
print("Creating ground-truth labels...")
df_anomalies = pd.read_csv(csv_path)
start_dt = pd.to_datetime(df_anomalies['anomaly_start'], utc=True)
end_dt = pd.to_datetime(df_anomalies['anomaly_end'], utc=True)
df_anomalies['start_ts'] = (start_dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
df_anomalies['end_ts'] = (end_dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')

df_features['known_anomaly'] = 0
for _, row in df_anomalies.iterrows():
    mask = (df_features['interval_start'] >= row['start_ts']) & (df_features['interval_start'] <= row['end_ts'])
    df_features.loc[mask, 'known_anomaly'] = 1

num_normal = (df_features['known_anomaly'] == 0).sum()
num_anom = (df_features['known_anomaly'] == 1).sum()
print(f"Labels - Normal: {num_normal}, Anomaly: {num_anom}")

window_coverage = []
for idx, row in df_anomalies.iterrows():
    count = ((df_features['interval_start'] >= row['start_ts']) & (df_features['interval_start'] <= row['end_ts'])).sum()
    window_coverage.append((row['number'], count))

print("Windows with zero telemetry intervals:")
for w, c in window_coverage:
    if c == 0:
        print(f"- {w}")

# TASK 4: Preprocess
print("Preprocessing features...")
df_features = df_features.fillna(0)

features_to_log = ['total_request_volume', 'client_request_volume', 'server_request_volume', 'error_request_volume', 'mean_recorded_avg_latency']
features_to_keep = ['active_hosts', 'active_endpoints']

X = pd.DataFrame()
for f in features_to_log:
    X[f + '_log1p'] = np.log1p(df_features[f].astype(float))
for f in features_to_keep:
    X[f] = df_features[f].astype(float)

# TASK 5 & 6: Primary Anomaly Detector
print("Running Isolation Forest experiments...")
contaminations = [0.01, 0.025, 0.05]
results_metrics = {}

for contam in contaminations:
    print(f"Testing contamination: {contam}")
    iso = IsolationForest(contamination=contam, random_state=42)
    preds = iso.fit_predict(X)
    pred_anom = (preds == -1).astype(int)
    scores = iso.decision_function(X) # lower score = more anomalous
    
    col_name = f'pred_anomaly_{contam}'
    score_name = f'score_{contam}'
    df_features[col_name] = pred_anom
    df_features[score_name] = scores
    
    # Interval level metrics
    y_true = df_features['known_anomaly']
    prec = precision_score(y_true, pred_anom, zero_division=0)
    rec = recall_score(y_true, pred_anom, zero_division=0)
    f1 = f1_score(y_true, pred_anom, zero_division=0)
    cm = confusion_matrix(y_true, pred_anom).tolist()
    
    # Window level detection
    detected_windows = 0
    for idx, row in df_anomalies.iterrows():
        w_start = row['start_ts'] - 300
        w_end = row['end_ts'] + 300
        w_mask = (df_features['interval_start'] >= w_start) & (df_features['interval_start'] <= w_end)
        
        if w_mask.sum() > 0:
            if df_features.loc[w_mask, col_name].sum() > 0:
                detected_windows += 1
                
    results_metrics[str(contam)] = {
        'interval_precision': float(prec),
        'interval_recall': float(rec),
        'interval_f1': float(f1),
        'confusion_matrix': cm,
        'detected_windows': detected_windows,
        'total_windows': len(df_anomalies),
        'windows_with_zero_telemetry': sum(1 for w, c in window_coverage if c == 0)
    }

print("Saving results...")
df_features.to_csv("outputs/anomaly_results.csv", index=False)
with open("outputs/evaluation_metrics.json", "w") as f:
    json.dump(results_metrics, f, indent=4)

# TASK 7: Visualizations
print("Generating plots...")

# Convert ts to datetime for plotting
df_features['timestamp'] = pd.to_datetime(df_features['interval_start'], unit='s', utc=True)

# 1. Total request volume
plt.figure(figsize=(15, 5))
plt.plot(df_features['timestamp'], df_features['total_request_volume'], color='blue', label='Total Volume', alpha=0.7)
for idx, row in df_anomalies.iterrows():
    plt.axvspan(pd.to_datetime(row['start_ts'], unit='s', utc=True), 
                pd.to_datetime(row['end_ts'], unit='s', utc=True), 
                color='red', alpha=0.3, label='Known Anomaly' if idx==0 else "")
plt.title("Total Request Volume Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/request_volume_timeline.png", dpi=150)
plt.close()

# 2. Mean recorded avg latency
plt.figure(figsize=(15, 5))
plt.plot(df_features['timestamp'], df_features['mean_recorded_avg_latency'], color='orange', label='Mean Recorded Avg Latency', alpha=0.7)
for idx, row in df_anomalies.iterrows():
    plt.axvspan(pd.to_datetime(row['start_ts'], unit='s', utc=True), 
                pd.to_datetime(row['end_ts'], unit='s', utc=True), 
                color='red', alpha=0.3, label='Known Anomaly' if idx==0 else "")
plt.title("Mean Recorded Average Latency Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/latency_timeline.png", dpi=150)
plt.close()

# 3. Anomaly score timeline (using 0.05 contamination for visualization)
best_contam = "0.05"
plt.figure(figsize=(15, 5))
plt.plot(df_features['timestamp'], df_features[f'score_{best_contam}'], color='green', label='Isolation Forest Score', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', label='Threshold')
for idx, row in df_anomalies.iterrows():
    plt.axvspan(pd.to_datetime(row['start_ts'], unit='s', utc=True), 
                pd.to_datetime(row['end_ts'], unit='s', utc=True), 
                color='red', alpha=0.3, label='Known Anomaly' if idx==0 else "")
# Mark detections
detections = df_features[df_features[f'pred_anomaly_{best_contam}'] == 1]
plt.scatter(detections['timestamp'], detections[f'score_{best_contam}'], color='red', label='Detected Anomaly', zorder=5, s=15)
plt.title(f"Anomaly Score Timeline (Contamination={best_contam})")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/anomaly_score_timeline.png", dpi=150)
plt.close()

# 4. Confusion matrix plot
cm = results_metrics[best_contam]['confusion_matrix']
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Contamination={best_contam})")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

end_time = time.time()
print(f"Pipeline finished successfully in {end_time - start_time:.2f} seconds.")
