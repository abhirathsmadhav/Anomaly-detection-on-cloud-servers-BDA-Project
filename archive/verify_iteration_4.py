import pandas as pd
import numpy as np

# 1. Correct Anomaly-Window Observability
print("--- 1. ANOMALY WINDOW OBSERVABILITY ---")
sys_df = pd.read_parquet('outputs/system_features.parquet')
intervals = sys_df['interval_start'].values

aw = pd.read_csv('anomaly_windows.csv')
aw['start_dt'] = pd.to_datetime(aw['anomaly_start'], utc=True)
aw['end_dt'] = pd.to_datetime(aw['anomaly_end'], utc=True)
aw['start_ts'] = (aw['start_dt'] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
aw['end_ts'] = (aw['end_dt'] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')

obs_windows = 0
zero_windows = 0

print("a-number | anomaly_start_utc | anomaly_end_utc | duration(s) | overlapping_buckets")
for idx, row in aw.iterrows():
    e_start = row['start_ts']
    e_end = row['end_ts']
    
    if e_start == e_end:
        # Point event
        overlaps = ((intervals <= e_start) & (e_start < intervals + 300)).sum()
    else:
        # Interval event
        overlaps = (np.maximum(intervals, e_start) < np.minimum(intervals + 300, e_end)).sum()
        
    if overlaps > 0:
        obs_windows += 1
    else:
        zero_windows += 1
        
    print(f"a{idx+1} | {row['start_dt']} | {row['end_dt']} | {e_end - e_start} | {overlaps}")

print(f"\nTotal documented windows: {len(aw)}")
print(f"Observable windows under exact bucket overlap: {obs_windows}")
print(f"Zero-observability windows: {zero_windows}")

# 3. Inspect system_features.parquet
print("\n--- 3. SYSTEM FEATURES INSPECTION ---")
print(f"Row count: {len(sys_df)}")
print(f"Columns: {list(sys_df.columns)}")

min_ts = sys_df['interval_start'].min()
max_ts = sys_df['interval_start'].max()
print(f"Min timestamp: {min_ts} ({pd.to_datetime(min_ts, unit='s', utc=True)})")
print(f"Max timestamp: {max_ts} ({pd.to_datetime(max_ts, unit='s', utc=True)})")

expected_intervals = (max_ts - min_ts) // 300 + 1
missing_intervals = expected_intervals - len(sys_df)
print(f"Missing intervals: {missing_intervals}")

print("\nNull counts:")
print(sys_df.isnull().sum())

print("\nBasic Statistics:")
cols_to_describe = [
    'total_request_volume', 'client_request_volume', 'server_request_volume',
    'active_hosts', 'active_endpoints', 'mean_recorded_avg_latency', 'error_request_volume'
]
# mean_recorded_avg_latency is named weighted_avg_latency in contextual output, let's check sys_df
existing_cols = [c for c in cols_to_describe if c in sys_df.columns]
print(sys_df[existing_cols].describe().T)

dt_series = pd.to_datetime(sys_df['interval_start'], unit='s', utc=True)
unique_dow = dt_series.dt.dayofweek.nunique()
unique_slots = ((dt_series.dt.hour * 60 + dt_series.dt.minute) // 5).nunique()

total_weeks = (max_ts - min_ts) / (86400 * 7)

print(f"\nUnique day-of-week values: {unique_dow}")
print(f"Unique 5-minute time-of-day slots: {unique_slots}")
print(f"Dataset spans {total_weeks:.2f} weeks.")
print(f"Enough weeks for seasonal baseline? {'Yes' if total_weeks > 2 else 'No'}")
