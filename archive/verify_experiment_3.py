import os, json, pandas as pd, numpy as np
from datetime import datetime

# 1. Output files
out_dir = 'outputs/'
files = os.listdir(out_dir)
print('--- 1. FILES ---')
for f in files:
    path = os.path.join(out_dir, f)
    size = os.path.getsize(path)
    mtime = os.path.getmtime(path)
    dt = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f'{f} | {size} bytes | {dt}')

# 2. Contextual result file
print('\n--- 2. CONTEXTUAL RESULTS ---')
df = pd.read_parquet('outputs/system_df_experiment_3.parquet')
print(f'Filename: system_df_experiment_3.parquet')
print(f'Row count: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Contains known_anomaly: {"known_anomaly" in df.columns}')
print(f'Contains Z>5 result (anom_count_5): {"anom_count_5" in df.columns}')
print(f'Contains anomalous_context_fraction: {"anomalous_context_fraction" in df.columns}')

# 3. Anomaly-window mapping
print('\n--- 3. ANOMALY WINDOW MAPPING ---')
df_anomalies = pd.read_csv('anomaly_windows.csv')
start_dt = pd.to_datetime(df_anomalies['anomaly_start'], utc=True)
end_dt = pd.to_datetime(df_anomalies['anomaly_end'], utc=True)
df_anomalies['start_ts'] = (start_dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
df_anomalies['end_ts'] = (end_dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')

total_windows = len(df_anomalies)
obs_windows = 0
zero_windows = 0

print(f'a-number | anomaly_start | anomaly_end | duration(s) | overlapping_intervals')
for idx, row in df_anomalies.iterrows():
    w_start = row['start_ts'] - 300
    w_end = row['end_ts'] + 300
    overlap = ((df['interval_start'] >= w_start) & (df['interval_start'] <= w_end)).sum()
    if overlap > 0:
        obs_windows += 1
    else:
        zero_windows += 1
    print(f'a{idx+1} | {row["anomaly_start"]} | {row["anomaly_end"]} | {row["end_ts"] - row["start_ts"]} | {overlap}')

print(f'Total documented windows: {total_windows}')
print(f'Total observable windows: {obs_windows}')
print(f'Number of zero-telemetry windows: {zero_windows}')

# 4. Metrics
print('\n--- 4. METRICS ---')
def print_metrics(name, pred, y_true, aw_df, df_sys):
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # fallback
        tn = ((y_true==0) & (pred==0)).sum()
        fp = ((y_true==0) & (pred==1)).sum()
        fn = ((y_true==1) & (pred==0)).sum()
        tp = ((y_true==1) & (pred==1)).sum()
        
    det = 0
    obs = 0
    for idx, row in aw_df.iterrows():
        w_start = row['start_ts'] - 300
        w_end = row['end_ts'] + 300
        w_mask = (df_sys['interval_start'] >= w_start) & (df_sys['interval_start'] <= w_end)
        if w_mask.sum() > 0:
            obs += 1
            if pred[w_mask].sum() > 0:
                det += 1
    
    print(f'{name}:')
    print(f'  Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%')
    print(f'  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}')
    print(f'  Pred count: {pred.sum()}')
    print(f'  Windows det / 25: {det}/25 | Windows det / {obs}: {det}/{obs}')

# A. IF
if os.path.exists('outputs/evaluation_metrics.json'):
    with open('outputs/evaluation_metrics.json', 'r') as f:
        metrics_if = json.load(f)
    print('A. IF baseline from evaluation_metrics.json')
    m = metrics_if['0.05']
    print(f"  Precision: {m['interval_precision']*100:.2f}% | Recall: {m['interval_recall']*100:.2f}% | F1: {m['interval_f1']*100:.2f}%")
    print(f"  Windows det / 25: {m['detected_windows']}/{m['total_windows']}")
    
# Contextual
y_true = df['known_anomaly']
print_metrics('B. Contextual Z > 5 ANY', (df['anom_count_5'] > 0).astype(int), y_true, df_anomalies, df)
print_metrics('C. Contextual Z > 5 (>= 0.5%)', (df['anomalous_context_fraction'] >= 0.005).astype(int), y_true, df_anomalies, df)
print_metrics('D. Contextual Z > 5 (>= 1%)', (df['anomalous_context_fraction'] >= 0.01).astype(int), y_true, df_anomalies, df)
print_metrics('E. Contextual Z > 5 (>= 2%)', (df['anomalous_context_fraction'] >= 0.02).astype(int), y_true, df_anomalies, df)
print_metrics('F. Contextual Z > 5 (>= 5%)', (df['anomalous_context_fraction'] >= 0.05).astype(int), y_true, df_anomalies, df)
print_metrics('G. Contextual Z > 5 (>= 10%)', (df['anomalous_context_fraction'] >= 0.10).astype(int), y_true, df_anomalies, df)

# 5. Distribution Finding
print('\n--- 5. DISTRIBUTION ---')
def print_dist(name, series):
    print(f'{name}:')
    print(f'  count: {series.count()}')
    print(f'  mean: {series.mean():.5f}')
    print(f'  median: {series.median():.5f}')
    print(f'  90th: {series.quantile(0.90):.5f}')
    print(f'  95th: {series.quantile(0.95):.5f}')
    print(f'  99th: {series.quantile(0.99):.5f}')
    print(f'  99.5th: {series.quantile(0.995):.5f}')
    print(f'  99.9th: {series.quantile(0.999):.5f}')
    print(f'  max: {series.max():.5f}')

print_dist('KNOWN ANOMALY INTERVALS', df[df['known_anomaly']==1]['anomalous_context_fraction'])
print_dist('NORMAL INTERVALS', df[df['known_anomaly']==0]['anomalous_context_fraction'])
