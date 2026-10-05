import duckdb
import pandas as pd
import numpy as np

print("Generating Step 10 stats...")

con = duckdb.connect()

aw = pd.read_csv('anomaly_windows.csv')
aw['start_dt'] = pd.to_datetime(aw['anomaly_start'], utc=True)
aw['end_dt'] = pd.to_datetime(aw['anomaly_end'], utc=True)
aw['start_ts'] = (aw['start_dt'] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
aw['end_ts'] = (aw['end_dt'] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')

# Create a temporary table with intervals and their labels
con.execute("""
CREATE TABLE interval_labels AS
SELECT DISTINCT interval_start, 0 as known_anomaly
FROM 'outputs/final_context_anomaly_scores.parquet'
""")

intervals_df = con.execute("SELECT interval_start FROM interval_labels").df()
intervals = intervals_df['interval_start'].values

for idx, row in aw.iterrows():
    e_start = row['start_ts']
    e_end = row['end_ts']
    if e_start == e_end:
        mask = (intervals <= e_start) & (e_start < intervals + 300)
    else:
        mask = np.maximum(intervals, e_start) < np.minimum(intervals + 300, e_end)
    
    anom_intervals = intervals[mask]
    if len(anom_intervals) > 0:
        anom_list = ",".join(map(str, anom_intervals))
        con.execute(f"UPDATE interval_labels SET known_anomaly = 1 WHERE interval_start IN ({anom_list})")

print("Calculating mean/median context score...")
res = con.execute("""
SELECT 
    l.known_anomaly,
    AVG(c.context_score) as mean_score,
    approx_quantile(c.context_score, 0.5) as median_score
FROM 'outputs/final_context_anomaly_scores.parquet' c
JOIN interval_labels l ON c.interval_start = l.interval_start
GROUP BY l.known_anomaly
""").df()
print(res)

print("\nTop anomalous contexts (highest context scores overall):")
top_contexts = con.execute("""
SELECT location, kind, host, endpoint, MAX(context_score) as max_score
FROM 'outputs/final_context_anomaly_scores.parquet'
GROUP BY location, kind, host, endpoint
ORDER BY max_score DESC
LIMIT 10
""").df()
print(top_contexts)
