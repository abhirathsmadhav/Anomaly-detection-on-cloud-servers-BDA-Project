import duckdb
import time
import os

start_time = time.time()
parquet_path = 'unpivoted_data.parquet'

# Configure DuckDB to avoid OOM
con = duckdb.connect()
con.execute("PRAGMA temp_directory='duckdb_tmp'")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='4GB'")

query = f'''
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
'''

print('Running DuckDB query with temp storage...')
try:
    df = con.execute(query).df()
    print(f'Done in {time.time() - start_time:.2f}s. Rows: {len(df)}')
    df.to_parquet('outputs/contextual_base.parquet', index=False)
except Exception as e:
    print(f"Error: {e}")
