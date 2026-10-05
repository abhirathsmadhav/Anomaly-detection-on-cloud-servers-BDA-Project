import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

print("STEP 1 - LOAD AND PREPARE")
df = pd.read_parquet('outputs/system_features.parquet')

# Convert to UTC datetime
df['dt'] = pd.to_datetime(df['interval_start'], unit='s', utc=True)
df['day_of_week'] = df['dt'].dt.dayofweek
df['time_of_day_slot'] = (df['dt'].dt.hour * 60 + df['dt'].dt.minute) // 5

primary_features = [
    'total_request_volume', 'client_request_volume', 'server_request_volume',
    'mean_recorded_avg_latency', 'error_request_volume', 'active_endpoints'
]
# Apply log1p to non-negative highly skewed features
# Endpoint is counts, log1p is fine too.
for col in primary_features:
    df[f'{col}_log'] = np.log1p(df[col].fillna(0))

log_features = [f'{col}_log' for col in primary_features]

print("STEP 2 - BUILD SEASONAL BASELINES")
def get_mad(x):
    return np.median(np.abs(x - np.median(x)))

# Overall MAD as fallback
fallback_mad = {}
for col in log_features:
    overall_mad = get_mad(df[col])
    # if overall_mad is 0, use a small epsilon
    fallback_mad[col] = overall_mad if overall_mad > 1e-6 else 1e-3

seasonal_stats = df.groupby(['day_of_week', 'time_of_day_slot'])[log_features].agg(['median', get_mad])
seasonal_stats.columns = [f"{col[0]}_{'median' if col[1]=='median' else 'mad'}" for col in seasonal_stats.columns]

df = df.merge(seasonal_stats.reset_index(), on=['day_of_week', 'time_of_day_slot'], how='left')

rz_features = []
for col in log_features:
    rz_col = f"rz_{col.replace('_log', '')}"
    rz_features.append(rz_col)
    
    mad_col = f"{col}_mad"
    med_col = f"{col}_median"
    
    # Fallback to overall MAD if seasonal MAD is 0
    safe_mad = np.where(df[mad_col] > 1e-6, df[mad_col], fallback_mad[col])
    
    df[rz_col] = (df[col] - df[med_col]) / (1.4826 * safe_mad)

print("STEP 3 - METHOD A: ROBUST SEASONAL ENSEMBLE")
# Max absolute robust deviation
df['max_abs_rz'] = df[rz_features].abs().max(axis=1)

print("STEP 4 - METHOD B: ISOLATION FOREST ON RESIDUAL FEATURES")
iso_025 = IsolationForest(contamination=0.025, random_state=42, n_jobs=-1)
iso_050 = IsolationForest(contamination=0.050, random_state=42, n_jobs=-1)

# Fit and predict (returns 1 for inliers, -1 for outliers)
print("Fitting IF 0.025...")
df['if_pred_025'] = iso_025.fit_predict(df[rz_features])
print("Fitting IF 0.050...")
df['if_pred_050'] = iso_050.fit_predict(df[rz_features])

# Convert -1 to 1 (anomaly), 1 to 0 (normal)
df['if_anom_025'] = (df['if_pred_025'] == -1).astype(int)
df['if_anom_050'] = (df['if_pred_050'] == -1).astype(int)

print("STEP 5 - CORRECT GROUND-TRUTH MAPPING")
aw = pd.read_csv('anomaly_windows.csv')
aw['start_dt'] = pd.to_datetime(aw['anomaly_start'], utc=True)
aw['end_dt'] = pd.to_datetime(aw['anomaly_end'], utc=True)
aw['start_ts'] = (aw['start_dt'] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
aw['end_ts'] = (aw['end_dt'] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')

df['known_anomaly'] = 0

intervals = df['interval_start'].values
for idx, row in aw.iterrows():
    e_start = row['start_ts']
    e_end = row['end_ts']
    
    if e_start == e_end:
        mask = (intervals <= e_start) & (e_start < intervals + 300)
    else:
        mask = np.maximum(intervals, e_start) < np.minimum(intervals + 300, e_end)
        
    df.loc[mask, 'known_anomaly'] = 1

y_true = df['known_anomaly']

print("STEP 6 - EVALUATION")
def evaluate_detector(name, pred_anom, y_true, df_sys, aw_df):
    acc = accuracy_score(y_true, pred_anom)
    prec = precision_score(y_true, pred_anom, zero_division=0)
    rec = recall_score(y_true, pred_anom, zero_division=0)
    f1 = f1_score(y_true, pred_anom, zero_division=0)
    cm = confusion_matrix(y_true, pred_anom, labels=[0, 1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = ((y_true==0) & (pred_anom==0)).sum()
        fp = ((y_true==0) & (pred_anom==1)).sum()
        fn = ((y_true==1) & (pred_anom==0)).sum()
        tp = ((y_true==1) & (pred_anom==1)).sum()
        
    det_windows = 0
    intervals_sys = df_sys['interval_start'].values
    pred_vals = pred_anom.values if isinstance(pred_anom, pd.Series) else pred_anom
    for idx, row in aw_df.iterrows():
        e_start = row['start_ts']
        e_end = row['end_ts']
        
        if e_start == e_end:
            w_mask = (intervals_sys <= e_start) & (e_start < intervals_sys + 300)
        else:
            w_mask = np.maximum(intervals_sys, e_start) < np.minimum(intervals_sys + 300, e_end)
            
        if w_mask.sum() > 0:
            if pred_vals[w_mask].sum() > 0:
                det_windows += 1
                
    return {
        "Name": name,
        "Accuracy": f"{acc*100:.2f}%",
        "Precision": f"{prec*100:.2f}%",
        "Recall": f"{rec*100:.2f}%",
        "F1": f"{f1*100:.2f}%",
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Pred_Anom_Count": pred_anom.sum(),
        "Windows_Detected": f"{det_windows}/25"
    }

results = []

# Method A candidates
thresholds_z = [3.0, 4.0, 5.0]
for t in thresholds_z:
    pred = (df['max_abs_rz'] > t).astype(int)
    results.append(evaluate_detector(f"Seasonal Ensemble Z > {t}", pred, y_true, df, aw))

# Method B candidates
results.append(evaluate_detector("Isolation Forest IF 0.025", df['if_anom_025'], y_true, df, aw))
results.append(evaluate_detector("Isolation Forest IF 0.050", df['if_anom_050'], y_true, df, aw))

print("\n--- ITERATION 4 RESULTS ---")
for r in results:
    print(f"{r['Name']}:")
    print(f"  Acc: {r['Accuracy']} | Prec: {r['Precision']} | Rec: {r['Recall']} | F1: {r['F1']}")
    print(f"  TP: {r['TP']} | FP: {r['FP']} | TN: {r['TN']} | FN: {r['FN']}")
    print(f"  Predicted Anomalies: {r['Pred_Anom_Count']} | Windows Detected: {r['Windows_Detected']}")

# Save
with open('outputs/experiment_4_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("Saved experiment_4_results.json")
