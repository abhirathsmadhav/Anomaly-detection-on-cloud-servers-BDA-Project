import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

print("Loading 5XX features...")
df = pd.read_parquet('outputs/5xx_features.parquet')
df_pivot = df.set_index('interval_start')
X_transformed = np.log1p(df_pivot)

print("Splitting Train/Test...")
interval_series = pd.to_datetime(df_pivot.index, unit='s', utc=True)
train_mask = (interval_series >= '2024-01-26') & (interval_series <= '2024-02-29 23:59:59')
test_mask = (interval_series >= '2024-03-01') & (interval_series <= '2024-05-31 23:59:59')

X_train_df = X_transformed[train_mask]
X_test_df = X_transformed[test_mask]

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)

print("Loading trained Autoencoder...")
autoencoder = joblib.load('outputs/final_ann_autoencoder.joblib')

print("Predicting to get MSE...")
X_train_pred = autoencoder.predict(X_train_scaled)
train_mse = pd.Series(np.mean(np.square(X_train_scaled - X_train_pred), axis=1), index=df_pivot.index[train_mask])

X_test_pred = autoencoder.predict(X_test_scaled)
test_mse = pd.Series(np.mean(np.square(X_test_scaled - X_test_pred), axis=1), index=df_pivot.index[test_mask])

print("Calculating Temporal Metrics...")
methods = {}

# 1. Baseline
methods['Baseline'] = {
    'train': train_mse,
    'test': test_mse
}

# 2. Rolling mean 3
methods['RM_3'] = {
    'train': train_mse.rolling(window=3, min_periods=1).mean(),
    'test': test_mse.rolling(window=3, min_periods=1).mean()
}

# 3. Rolling mean 6
methods['RM_6'] = {
    'train': train_mse.rolling(window=6, min_periods=1).mean(),
    'test': test_mse.rolling(window=6, min_periods=1).mean()
}

# 4. Rolling median 3
methods['RMed_3'] = {
    'train': train_mse.rolling(window=3, min_periods=1).median(),
    'test': test_mse.rolling(window=3, min_periods=1).median()
}

print("Loading Ground Truth...")
aw = pd.read_csv('anomaly_windows.csv')
aw['start_dt'] = pd.to_datetime(aw['anomaly_start'], utc=True)
aw['end_dt'] = pd.to_datetime(aw['anomaly_end'], utc=True)
aw['start_ts'] = (aw['start_dt'] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
aw['end_ts'] = (aw['end_dt'] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')

aw_test = aw[(aw['start_dt'] >= '2024-03-01') & (aw['start_dt'] <= '2024-05-31 23:59:59')].copy()
test_intervals = test_mse.index.values

known_anomaly = np.zeros(len(test_intervals), dtype=int)
observable_windows = 0
for idx, row in aw_test.iterrows():
    e_start = row['start_ts']
    e_end = row['end_ts']
    if e_start == e_end:
        mask = (test_intervals <= e_start) & (e_start < test_intervals + 300)
    else:
        mask = np.maximum(test_intervals, e_start) < np.minimum(test_intervals + 300, e_end)
    
    if mask.sum() > 0:
        observable_windows += 1
        known_anomaly[mask] = 1

y_true = known_anomaly

def evaluate(pred_series, name, thresh):
    pred = (pred_series >= thresh).astype(int)
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    det_windows = 0
    pred_vals = pred.values
    for idx, row in aw_test.iterrows():
        e_start = row['start_ts']
        e_end = row['end_ts']
        if e_start == e_end:
            w_mask = (test_intervals <= e_start) & (e_start < test_intervals + 300)
        else:
            w_mask = np.maximum(test_intervals, e_start) < np.minimum(test_intervals + 300, e_end)
        if w_mask.sum() > 0 and pred_vals[w_mask].sum() > 0:
            det_windows += 1
            
    return {
        "Method": name,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
        "Pred_Anomalous_Intervals": pred.sum(),
        "Windows_Detected": det_windows,
        "Threshold": thresh
    }

results = []
test_out_df = pd.DataFrame({'interval_start': test_intervals, 'known_anomaly': known_anomaly})
test_out_df['timestamp'] = pd.to_datetime(test_intervals, unit='s', utc=True)

for m_name, m_data in methods.items():
    thresh = np.percentile(m_data['train'].dropna(), 99.9)
    res = evaluate(m_data['test'], m_name, thresh)
    results.append(res)
    test_out_df[f"{m_name}_score"] = m_data['test'].values
    test_out_df[f"{m_name}_pred"] = (m_data['test'].values >= thresh).astype(int)

res_df = pd.DataFrame(results)
print("\n--- TEMPORAL REFINEMENT RESULTS ---")
print(res_df.drop(columns=['Threshold']).to_string())

best_method = max(results, key=lambda x: x['F1'])
print(f"\nBest Method based on F1: {best_method['Method']}")

# Save outputs
test_out_df.to_parquet('outputs/final_ann_refined_interval_results.parquet', index=False)
res_df.to_csv('outputs/final_ann_refined_comparison.csv', index=False)
with open('outputs/final_ann_refined_results.json', 'w') as f:
    json.dump(results, f, indent=4, cls=NpEncoder)

if best_method['Method'] != 'Baseline':
    b_name = best_method['Method']
    plt.figure(figsize=(15,5))
    plt.plot(test_out_df['timestamp'], test_out_df[f"{b_name}_score"], color='purple', label=f"{b_name} Score")
    anoms = test_out_df[test_out_df[f"{b_name}_pred"] == 1]
    plt.scatter(anoms['timestamp'], anoms[f"{b_name}_score"], color='red', label='Predicted Anomaly')
    for idx, row in aw_test.iterrows():
        plt.axvspan(row['start_dt'], row['end_dt'], color='red', alpha=0.3)
    plt.axhline(best_method['Threshold'], color='black', linestyle='--', label='99.9th Train Pct')
    plt.title(f'Refined Anomaly Score over Test Period ({b_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/final_refined_score_timeline.png', dpi=150)
    plt.close()
    
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_true, test_out_df[f"{b_name}_pred"], labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix ({b_name})')
    plt.savefig('outputs/final_refined_confusion_matrix.png', dpi=150)
    plt.close()
    print("Generated visual plots for better performing refined method.")
else:
    print("Baseline remained the best method. No new plots generated.")
