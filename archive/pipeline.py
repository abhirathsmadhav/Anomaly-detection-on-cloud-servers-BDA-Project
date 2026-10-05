import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

start_time = time.time()
os.makedirs("outputs", exist_ok=True)

print("Starting Spark Session...")
spark = SparkSession.builder \
    .appName("AnomalyDetectionPipeline") \
    .config("spark.driver.memory", "10g") \
    .config("spark.executor.memory", "10g") \
    .config("spark.sql.shuffle.partitions", "20") \
    .config("spark.memory.offHeap.enabled","true") \
    .config("spark.memory.offHeap.size","2g") \
    .getOrCreate()

parquet_path = "unpivoted_data.parquet"
csv_path = "anomaly_windows.csv"

print("Loading raw parquet data...")
raw_df = spark.read.parquet(parquet_path)

# Task 1: Calculate system-wide features
print("Calculating system-wide features via PySpark...")
sys_features = raw_df.groupBy("interval_start").agg(
    F.sum(F.when(F.col("aggregated_stats_name") == "count", F.col("aggregated_stats_value")).otherwise(0)).alias("total_request_volume"),
    F.sum(F.when((F.col("kind") == "CLIENT") & (F.col("aggregated_stats_name") == "count"), F.col("aggregated_stats_value")).otherwise(0)).alias("client_request_volume"),
    F.sum(F.when((F.col("kind") == "SERVER") & (F.col("aggregated_stats_name") == "count"), F.col("aggregated_stats_value")).otherwise(0)).alias("server_request_volume"),
    F.countDistinct("host").alias("active_hosts"),
    F.countDistinct("endpoint").alias("active_endpoints"),
    F.avg(F.when(F.col("aggregated_stats_name") == "avg", F.col("aggregated_stats_value"))).alias("mean_recorded_avg_latency"),
    F.sum(F.when((F.col("statusCode") >= 500) & (F.col("aggregated_stats_name") == "count"), F.col("aggregated_stats_value")).otherwise(0)).alias("error_request_volume")
)

print("Collecting features to Pandas...")
df_features = sys_features.toPandas()
spark.stop()

# Sort by time
df_features = df_features.sort_values("interval_start").reset_index(drop=True)

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
# Missing timestamps are intentionally NOT imputed or marked automatically as anomalies 
# because there are only 15 missing ticks over 4 months, which doesn't disrupt the macro-temporal flow.

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
# Fill nulls in aggregations
df_features = df_features.fillna(0)

features_to_log = ['total_request_volume', 'client_request_volume', 'server_request_volume', 'error_request_volume', 'mean_recorded_avg_latency']
features_to_keep = ['active_hosts', 'active_endpoints']

X = pd.DataFrame()
for f in features_to_log:
    # Use float to avoid type issues with PySpark results
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
