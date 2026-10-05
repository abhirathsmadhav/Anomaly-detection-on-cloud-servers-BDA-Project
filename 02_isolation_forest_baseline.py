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

con = duckdb.connect()
con.execute("PRAGMA temp_directory='duckdb_tmp'")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='4GB'")

print("STEP 2 - CORRECT CONTEXT AGGREGATION")
parquet_path = 'unpivoted_data.parquet'

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

print("\nSTEP 3 - CONTEXT FILTERING")
con.execute("""
CREATE TABLE retained_contexts AS
SELECT location, kind, host, endpoint, COUNT(*) as obs_count
FROM contextual_raw
GROUP BY location, kind, host, endpoint
HAVING COUNT(*) >= 100
""")

print("\nSTEP 4 - CONTEXT-LEVEL BASELINE")
print("Filtering logs...")
con.execute("""
CREATE TABLE filtered_logs AS
SELECT 
    c.interval_start, c.location, c.kind, c.host, c.endpoint,
    LN(1 + COALESCE(c.total_request_volume, 0)) as trv_log,
    LN(1 + COALESCE(c.weighted_avg_latency, 0)) as wal_log,
    LN(1 + COALESCE(c.error_request_volume, 0)) as erv_log
FROM contextual_raw c
JOIN retained_contexts r
ON c.location = r.location AND c.kind = r.kind AND c.host = r.host AND c.endpoint = r.endpoint
""")

print("Calculating Context Medians...")
con.execute("""
CREATE TABLE context_medians AS
SELECT location, kind, host, endpoint,
    approx_quantile(trv_log, 0.5) as trv_med,
    approx_quantile(wal_log, 0.5) as wal_med,
    approx_quantile(erv_log, 0.5) as erv_med
FROM filtered_logs
GROUP BY location, kind, host, endpoint
""")

print("Calculating Context MADs...")
con.execute("""
CREATE TABLE context_mads AS
SELECT f.location, f.kind, f.host, f.endpoint,
    approx_quantile(ABS(f.trv_log - m.trv_med), 0.5) as trv_mad,
    approx_quantile(ABS(f.wal_log - m.wal_med), 0.5) as wal_mad,
    approx_quantile(ABS(f.erv_log - m.erv_med), 0.5) as erv_mad
FROM filtered_logs f
JOIN context_medians m ON f.location=m.location AND f.kind=m.kind AND f.host=m.host AND f.endpoint=m.endpoint
GROUP BY f.location, f.kind, f.host, f.endpoint
""")

print("Calculating Global MADs for Fallback...")
g_med = con.execute("""
SELECT 
    approx_quantile(trv_log, 0.5), 
    approx_quantile(wal_log, 0.5), 
    approx_quantile(erv_log, 0.5) 
FROM filtered_logs
""").fetchone()

g_mad = con.execute(f"""
SELECT 
    approx_quantile(ABS(trv_log - {g_med[0]}), 0.5), 
    approx_quantile(ABS(wal_log - {g_med[1]}), 0.5), 
    approx_quantile(ABS(erv_log - {g_med[2]}), 0.5) 
FROM filtered_logs
""").fetchone()

trv_fb = g_mad[0] if g_mad[0] > 1e-6 else 1e-3
wal_fb = g_mad[1] if g_mad[1] > 1e-6 else 1e-3
erv_fb = g_mad[2] if g_mad[2] > 1e-6 else 1e-3

print("\nSTEP 5 - CONTEXT ANOMALY SCORE")
print("Calculating Robust Z-Scores and Context Scores...")
con.execute(f"""
CREATE TABLE context_scores AS
SELECT 
    f.interval_start, f.location, f.kind, f.host, f.endpoint,
    ABS(f.trv_log - m.trv_med) / (1.4826 * CASE WHEN md.trv_mad > 1e-6 THEN md.trv_mad ELSE {trv_fb} END) as trv_z,
    ABS(f.wal_log - m.wal_med) / (1.4826 * CASE WHEN md.wal_mad > 1e-6 THEN md.wal_mad ELSE {wal_fb} END) as wal_z,
    ABS(f.erv_log - m.erv_med) / (1.4826 * CASE WHEN md.erv_mad > 1e-6 THEN md.erv_mad ELSE {erv_fb} END) as erv_z,
    GREATEST(
        ABS(f.trv_log - m.trv_med) / (1.4826 * CASE WHEN md.trv_mad > 1e-6 THEN md.trv_mad ELSE {trv_fb} END),
        ABS(f.wal_log - m.wal_med) / (1.4826 * CASE WHEN md.wal_mad > 1e-6 THEN md.wal_mad ELSE {wal_fb} END),
        ABS(f.erv_log - m.erv_med) / (1.4826 * CASE WHEN md.erv_mad > 1e-6 THEN md.erv_mad ELSE {erv_fb} END)
    ) as context_score
FROM filtered_logs f
JOIN context_medians m ON f.location=m.location AND f.kind=m.kind AND f.host=m.host AND f.endpoint=m.endpoint
JOIN context_mads md ON f.location=md.location AND f.kind=md.kind AND f.host=md.host AND f.endpoint=md.endpoint
""")

print("\nSTEP 6 - CONVERT CONTEXT SCORES TO INTERVAL SCORES")
# Calculating Top 5 context score in SQL using window functions
con.execute("""
CREATE TABLE interval_scores AS
WITH ranked AS (
    SELECT 
        interval_start, 
        context_score,
        ROW_NUMBER() OVER (PARTITION BY interval_start ORDER BY context_score DESC) as rnk
    FROM context_scores
)
SELECT 
    interval_start,
    MAX(context_score) as max_context_score,
    AVG(CASE WHEN rnk <= 5 THEN context_score ELSE NULL END) as top5_context_score,
    SUM(CASE WHEN context_score >= 3 THEN 1 ELSE 0 END) as anomalous_context_count
FROM ranked
GROUP BY interval_start
""")

print("Saving final context_scores to parquet...")
con.execute("COPY context_scores TO 'outputs/final_context_anomaly_scores.parquet' (FORMAT PARQUET)")

print("Fetching interval scores to Pandas...")
interval_df = con.execute("SELECT * FROM interval_scores").df()

# Also calculate total system volume for plotting
sys_vol = con.execute("""
SELECT interval_start, SUM(COALESCE(total_request_volume, 0)) as total_sys_volume
FROM contextual_raw
GROUP BY interval_start
""").df()
interval_df = interval_df.merge(sys_vol, on='interval_start', how='left')

con.close()

print("\n--- Aggregation Stats ---")
print(f"Row count: {len(interval_df)}")

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
# the context anomaly scores were already saved inside duckdb query to parquet
res_df = pd.DataFrame(results)
res_df.to_csv('outputs/final_context_results.csv', index=False)

with open('outputs/final_context_results.json', 'w') as f:
    json.dump(results, f, indent=4, cls=NpEncoder)

print(f"Total execution time: {time.time() - start_time:.2f} seconds")
