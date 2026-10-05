import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

start_time = time.time()
os.makedirs("outputs", exist_ok=True)

parquet_path = "unpivoted_data.parquet"
csv_path = "anomaly_windows.csv"

# ==================================================
# STEP 1: BUILD CONTEXTUAL FEATURES
# ==================================================
print("Calculating contextual features via DuckDB...")
con = duckdb.connect()

query = f"""
WITH pivoted_raw AS (
    SELECT 
        interval_start, location, kind, host, method, statusCode, endpoint,
        MAX(CASE WHEN aggregated_stats_name = 'count' THEN aggregated_stats_value END) AS count_val,
        MAX(CASE WHEN aggregated_stats_name = 'avg' THEN aggregated_stats_value END) AS avg_val,
        MAX(CASE WHEN aggregated_stats_name = 'min' THEN aggregated_stats_value END) AS min_val,
        MAX(CASE WHEN aggregated_stats_name = 'max' THEN aggregated_stats_value END) AS max_val
    FROM '{parquet_path}'
    GROUP BY interval_start, location, kind, host, method, statusCode, endpoint
)
SELECT
    interval_start, host, endpoint,
    SUM(count_val) AS total_request_volume,
    SUM(CASE WHEN kind = 'CLIENT' THEN count_val ELSE 0 END) AS client_request_volume,
    SUM(CASE WHEN kind = 'SERVER' THEN count_val ELSE 0 END) AS server_request_volume,
    SUM(CASE WHEN statusCode >= 500 THEN count_val ELSE 0 END) AS error_request_volume,
    SUM(avg_val * count_val) / NULLIF(SUM(count_val), 0) AS weighted_avg_latency,
    MIN(min_val) AS min_latency,
    MAX(max_val) AS max_latency,
    COUNT(DISTINCT statusCode) AS active_status_codes,
    COUNT(DISTINCT method) AS active_methods
FROM pivoted_raw
GROUP BY interval_start, host, endpoint
"""

print("Executing SQL Pivot and Grouping (this will process 413M rows)...")
df = con.execute(query).df()
print(f"Contextual features generated: {len(df)} rows.")

# ==================================================
# STEP 2: CONTEXT BASELINE
# ==================================================
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

# ==================================================
# STEP 3: CONTEXTUAL ANOMALY SCORE
# ==================================================
print("Calculating contextual anomaly scores...")
for f in features:
    df[f'{f}_z'] = np.abs(df[f] - df[f'{f}_median']) / (1.4826 * df[f'{f}_mad'] + 1e-6)

df['context_anomaly_score'] = df[[f'{f}_z' for f in features]].max(axis=1)

# ==================================================
# STEP 4: RETURN TO SYSTEM TIMELINE
# ==================================================
print("Aggregating back to system-wide timeline...")
thresholds = [3.0, 4.0, 5.0]

for t in thresholds:
    df[f'is_anom_{t}'] = (df['context_anomaly_score'] > t).astype(int)

system_df = df.groupby('interval_start').agg(
    max_context_anomaly_score=('context_anomaly_score', 'max'),
    num_contexts=('host', 'count'),
    anom_count_3=('is_anom_3.0', 'sum'),
    anom_count_4=('is_anom_4.0', 'sum'),
    anom_count_5=('is_anom_5.0', 'sum'),
    total_request_volume=('total_request_volume', 'sum')
).reset_index()

system_df['anomalous_context_fraction_3'] = system_df['anom_count_3'] / system_df['num_contexts']
system_df['anomalous_context_fraction_4'] = system_df['anom_count_4'] / system_df['num_contexts']
system_df['anomalous_context_fraction_5'] = system_df['anom_count_5'] / system_df['num_contexts']

# Add known anomalies for evaluation
df_anomalies = pd.read_csv(csv_path)
start_dt = pd.to_datetime(df_anomalies['anomaly_start'], utc=True)
end_dt = pd.to_datetime(df_anomalies['anomaly_end'], utc=True)
df_anomalies['start_ts'] = (start_dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
df_anomalies['end_ts'] = (end_dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')

system_df['known_anomaly'] = 0
for _, row in df_anomalies.iterrows():
    mask = (system_df['interval_start'] >= row['start_ts']) & (system_df['interval_start'] <= row['end_ts'])
    system_df.loc[mask, 'known_anomaly'] = 1

# ==================================================
# STEP 5: THRESHOLD EXPERIMENTS
# ==================================================
print("Evaluating threshold experiments...")
results_metrics = {}

for t in thresholds:
    # A system interval is an anomaly if AT LEAST ONE context is an anomaly
    pred_anom = (system_df[f'anom_count_{int(t)}'] > 0).astype(int)
    
    y_true = system_df['known_anomaly']
    prec = precision_score(y_true, pred_anom, zero_division=0)
    rec = recall_score(y_true, pred_anom, zero_division=0)
    f1 = f1_score(y_true, pred_anom, zero_division=0)
    
    detected_windows = 0
    for idx, row in df_anomalies.iterrows():
        w_start = row['start_ts'] - 300
        w_end = row['end_ts'] + 300
        w_mask = (system_df['interval_start'] >= w_start) & (system_df['interval_start'] <= w_end)
        if w_mask.sum() > 0 and pred_anom[w_mask].sum() > 0:
            detected_windows += 1
            
    results_metrics[str(t)] = {
        'interval_precision': float(prec),
        'interval_recall': float(rec),
        'interval_f1': float(f1),
        'detected_windows': detected_windows,
        'total_windows': len(df_anomalies)
    }

print("\n--- RESULTS COMPARISON ---")
print("Method | Precision | Recall | F1 | Windows Detected")
print("Baseline (IF 0.05) | 2.18% | 4.48% | 2.94% | 8/25")
for t in thresholds:
    res = results_metrics[str(t)]
    print(f"Contextual (Z>{t}) | {res['interval_precision']*100:.2f}% | {res['interval_recall']*100:.2f}% | {res['interval_f1']*100:.2f}% | {res['detected_windows']}/25")

# ==================================================
# STEP 7: VISUALIZATION
# ==================================================
print("Generating contextual plots...")
system_df['timestamp'] = pd.to_datetime(system_df['interval_start'], unit='s', utc=True)

# 1. Max Contextual Anomaly Score Over Time
plt.figure(figsize=(15, 5))
plt.plot(system_df['timestamp'], system_df['max_context_anomaly_score'], color='purple', alpha=0.8, label='Max Context Robust Z-Score')
for idx, row in df_anomalies.iterrows():
    plt.axvspan(pd.to_datetime(row['start_ts'], unit='s', utc=True), 
                pd.to_datetime(row['end_ts'], unit='s', utc=True), 
                color='red', alpha=0.3, label='Known Anomaly' if idx==0 else "")
plt.axhline(5.0, color='black', linestyle='--', label='Threshold Z=5')
plt.title("Max Contextual Anomaly Score Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/context_max_score_timeline.png", dpi=150)
plt.close()

# 2. Number of Anomalous Contexts
plt.figure(figsize=(15, 5))
plt.plot(system_df['timestamp'], system_df['anom_count_5'], color='darkorange', alpha=0.8, label='Number of Anomalous Contexts (Z>5)')
for idx, row in df_anomalies.iterrows():
    plt.axvspan(pd.to_datetime(row['start_ts'], unit='s', utc=True), 
                pd.to_datetime(row['end_ts'], unit='s', utc=True), 
                color='red', alpha=0.3)
plt.title("Number of Anomalous Host/Endpoint Contexts Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/context_anomalous_counts.png", dpi=150)
plt.close()

with open("outputs/contextual_evaluation_metrics.json", "w") as f:
    json.dump(results_metrics, f, indent=4)

end_time = time.time()
print(f"Contextual pipeline finished successfully in {end_time - start_time:.2f} seconds.")
