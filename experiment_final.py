import duckdb
import pandas as pd
import numpy as np
import json
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

start_time = time.time()
os.makedirs('outputs', exist_ok=True)

print("STEP 2 - CORRECT CONTEXT AGGREGATION")
parquet_path = 'unpivoted_data.parquet'

con = duckdb.connect()
con.execute("PRAGMA temp_directory='duckdb_tmp'")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='4GB'")

print("Running DuckDB aggregation directly to table...")
query = f'''
CREATE TABLE contextual_raw AS
WITH pivoted AS (
    SELECT 
        interval_start, location, kind, host, method, statusCode, endpoint,
        MAX(CASE WHEN aggregated_stats_name = 'count' THEN aggregated_stats_value END) AS count_val,
        MAX(CASE WHEN aggregated_stats_name = 'avg' THEN aggregated_stats_value END) AS avg_val
    FROM '{parquet_path}'
    GROUP BY interval_start, location, kind, host, method, statusCode, endpoint
)
SELECT
    interval_start, location, kind, host, endpoint,
    SUM(count_val) AS total_request_volume,
    SUM(CASE WHEN statusCode >= 500 THEN count_val ELSE 0 END) AS error_request_volume,
    SUM(avg_val * count_val) / NULLIF(SUM(count_val), 0) AS weighted_avg_latency
FROM pivoted
GROUP BY interval_start, location, kind, host, endpoint
'''
con.execute(query)

print("Exporting complete aggregation to parquet...")
con.execute("COPY contextual_raw TO 'outputs/contextual_correct.parquet' (FORMAT PARQUET)")

total_rows = con.execute("SELECT COUNT(*) FROM contextual_raw").fetchone()[0]
unique_contexts = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT location, kind, host, endpoint FROM contextual_raw)").fetchone()[0]
min_ts = con.execute("SELECT MIN(interval_start) FROM contextual_raw").fetchone()[0]
max_ts = con.execute("SELECT MAX(interval_start) FROM contextual_raw").fetchone()[0]
nulls = con.execute("""
    SELECT 
        SUM(CASE WHEN total_request_volume IS NULL THEN 1 ELSE 0 END) as trv,
        SUM(CASE WHEN error_request_volume IS NULL THEN 1 ELSE 0 END) as erv,
        SUM(CASE WHEN weighted_avg_latency IS NULL THEN 1 ELSE 0 END) as wal
    FROM contextual_raw
""").fetchone()

print("\n--- Aggregation Stats ---")
print(f"Row count: {total_rows}")
print(f"Distinct contexts: {unique_contexts}")
print(f"Timestamp range: {min_ts} to {max_ts}")
print(f"Null counts: total_request_volume={nulls[0]}, error_request_volume={nulls[1]}, weighted_avg_latency={nulls[2]}")

print("\nSTEP 3 - CONTEXT FILTERING")
con.execute("""
CREATE TABLE retained_contexts AS
SELECT location, kind, host, endpoint, COUNT(*) as obs_count
FROM contextual_raw
GROUP BY location, kind, host, endpoint
HAVING COUNT(*) >= 100
""")

total_contexts = unique_contexts
retained_count = con.execute("SELECT COUNT(*) FROM retained_contexts").fetchone()[0]
print(f"Total contexts: {total_contexts}")
print(f"Retained contexts (>= 100 obs): {retained_count}")
print(f"Percentage retained: {retained_count / total_contexts * 100:.2f}%")

print("Pulling filtered data into Pandas...")
df = con.execute("""
SELECT c.* 
FROM contextual_raw c
JOIN retained_contexts r
ON c.location = r.location AND c.kind = r.kind AND c.host = r.host AND c.endpoint = r.endpoint
""").df()

con.close()

print("\nSTEP 4 - CONTEXT-LEVEL BASELINE")
for col in ['total_request_volume', 'weighted_avg_latency', 'error_request_volume']:
    df[f'{col}_log'] = np.log1p(df[col].fillna(0))

def get_mad(x):
    return np.median(np.abs(x - np.median(x)))

features_log = ['total_request_volume_log', 'weighted_avg_latency_log', 'error_request_volume_log']

print("Calculating baselines per context...")
baselines = df.groupby(['location', 'kind', 'host', 'endpoint'])[features_log].agg(['median', get_mad])
baselines.columns = [f"{c[0]}_{'median' if c[1]=='median' else 'mad'}" for c in baselines.columns]
df = df.merge(baselines.reset_index(), on=['location', 'kind', 'host', 'endpoint'], how='left')

print("Calculating robust deviations...")
global_mad = {col: get_mad(df[col]) for col in features_log}
fallback_mad = {col: global_mad[col] if global_mad[col] > 1e-6 else 1e-3 for col in features_log}

for col in features_log:
    med_col = f"{col}_median"
    mad_col = f"{col}_mad"
    safe_mad = np.where(df[mad_col] > 1e-6, df[mad_col], fallback_mad[col])
    df[f"{col}_z"] = np.abs(df[col] - df[med_col]) / (1.4826 * safe_mad)

print("\nSTEP 5 - CONTEXT ANOMALY SCORE")
z_cols = [f"{col}_z" for col in features_log]
df['context_score'] = df[z_cols].max(axis=1)

def top2_mean(row):
    sorted_vals = np.sort(row)[::-1]
    return np.mean(sorted_vals[:2])

df['context_top2_score'] = df[z_cols].apply(top2_mean, axis=1)

print("\nSTEP 6 - CONVERT CONTEXT SCORES TO INTERVAL SCORES")
def top5_mean(x):
    sorted_vals = np.sort(x)[::-1]
    return np.mean(sorted_vals[:5])

interval_df = df.groupby('interval_start').agg(
    max_context_score=('context_score', 'max'),
    top5_context_score=('context_score', top5_mean),
    anomalous_context_count=('context_score', lambda x: (x >= 3).sum()),
    total_sys_volume=('total_request_volume', 'sum')
).reset_index()

print("\nSTEP 8 - GROUND TRUTH")
aw = pd.read_csv('anomaly_windows.csv')
aw['start_dt'] = pd.to_datetime(aw['anomaly_start'], utc=True)
aw['end_dt'] = pd.to_datetime(aw['anomaly_end'], utc=True)
aw['start_ts'] = (aw['start_dt'] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
aw['end_ts'] = (aw['end_dt'] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')

interval_df['known_anomaly'] = 0
intervals = interval_df['interval_start'].values

for idx, row in aw.iterrows():
    e_start = row['start_ts']
    e_end = row['end_ts']
    if e_start == e_end:
        mask = (intervals <= e_start) & (e_start < intervals + 300)
    else:
        mask = np.maximum(intervals, e_start) < np.minimum(intervals + 300, e_end)
    interval_df.loc[mask, 'known_anomaly'] = 1

print("\nSTEP 7 & 9 - THRESHOLD TESTING AND EVALUATION")
y_true = interval_df['known_anomaly']

def evaluate(name, score_col, threshold):
    pred = (interval_df[score_col] >= threshold).astype(int)
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    det_windows = 0
    pred_vals = pred.values
    for idx, row in aw.iterrows():
        e_start = row['start_ts']
        e_end = row['end_ts']
        if e_start == e_end:
            w_mask = (intervals <= e_start) & (e_start < intervals + 300)
        else:
            w_mask = np.maximum(intervals, e_start) < np.minimum(intervals + 300, e_end)
        
        if w_mask.sum() > 0 and pred_vals[w_mask].sum() > 0:
            det_windows += 1
            
    return {
        "Method": name.split(' - ')[0],
        "Score": name.split(' - ')[1],
        "Threshold": threshold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Pred_Anom_Intervals": pred.sum(),
        "Windows_Detected": det_windows
    }

results = []
for t in [3.0, 4.0, 5.0]:
    results.append(evaluate("Max Context Score - max_context_score", "max_context_score", t))
    results.append(evaluate("Top 5 Context Score - top5_context_score", "top5_context_score", t))

best_res = max(results, key=lambda x: x['F1'])

print("\n--- COMPARISON TABLE ---")
print("Method | Score | Threshold | Precision | Recall | F1 | Accuracy | Pred Anomalous Intervals | Windows Det / 25")
for r in results:
    print(f"{r['Method']} | {r['Score']} | {r['Threshold']} | {r['Precision']*100:.2f}% | {r['Recall']*100:.2f}% | {r['F1']*100:.2f}% | {r['Accuracy']*100:.2f}% | {r['Pred_Anom_Intervals']} | {r['Windows_Detected']}/25")

print(f"\nBest Method: {best_res['Method']} {best_res['Score']} > {best_res['Threshold']}")

print("\nSTEP 10 - ADDITIONAL ANALYSIS")
df = df.merge(interval_df[['interval_start', 'known_anomaly']], on='interval_start', how='left')
best_score_col = best_res['Score']

norm_scores = df[df['known_anomaly'] == 0]['context_score']
anom_scores = df[df['known_anomaly'] == 1]['context_score']

print(f"Normal Intervals - Mean context_score: {norm_scores.mean():.4f}, Median: {norm_scores.median():.4f}")
print(f"Anomaly Intervals - Mean context_score: {anom_scores.mean():.4f}, Median: {anom_scores.median():.4f}")

print("Top 10 anomalous contexts (highest context scores overall):")
top_contexts = df.nlargest(10, 'context_score')[['location', 'kind', 'host', 'endpoint', 'context_score', 'known_anomaly']]
print(top_contexts.to_string())

print("\nSTEP 11 - VISUALIZATION")
interval_df['timestamp'] = pd.to_datetime(interval_df['interval_start'], unit='s', utc=True)

plt.figure(figsize=(15, 5))
plt.plot(interval_df['timestamp'], interval_df[best_res['Score']], color='blue', alpha=0.7, label=f"{best_res['Score']}")
for idx, row in aw.iterrows():
    plt.axvspan(row['start_dt'], row['end_dt'], color='red', alpha=0.3, label='Known Anomaly' if idx==0 else "")
plt.axhline(best_res['Threshold'], color='black', linestyle='--', label=f"Threshold = {best_res['Threshold']}")
plt.title(f"Interval Anomaly Score Timeline ({best_res['Score']})")
plt.legend()
plt.tight_layout()
plt.savefig('outputs/final_context_score_timeline.png', dpi=150)
plt.close()

plt.figure(figsize=(15, 5))
plt.plot(interval_df['timestamp'], interval_df['total_sys_volume'], color='gray', alpha=0.6, label='Total Sys Volume')
pred_mask = interval_df[best_res['Score']] >= best_res['Threshold']
plt.scatter(interval_df.loc[pred_mask, 'timestamp'], interval_df.loc[pred_mask, 'total_sys_volume'], color='red', s=10, label='Predicted Anomaly')
plt.title("Total Request Volume with Detected Anomalies")
plt.legend()
plt.tight_layout()
plt.savefig('outputs/final_request_volume_anomalies.png', dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_true, pred_mask, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title(f"Confusion Matrix ({best_res['Score']} >= {best_res['Threshold']})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('outputs/final_confusion_matrix.png', dpi=150)
plt.close()

print("\nSTEP 12 - FINAL OUTPUT")
interval_df.to_parquet('outputs/final_context_anomaly_scores.parquet', index=False)
res_df = pd.DataFrame(results)
res_df.to_csv('outputs/final_context_results.csv', index=False)

with open('outputs/final_context_results.json', 'w') as f:
    json.dump(results, f, indent=4, cls=NpEncoder)

print(f"Total execution time: {time.time() - start_time:.2f} seconds")
