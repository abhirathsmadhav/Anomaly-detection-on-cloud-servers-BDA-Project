# Anomaly Detection in Cloud Server Telemetry

## Project Objective
This project implements a system-wide anomaly detection pipeline for cloud server telemetry data. The objective is to evaluate whether aggregated, system-level features extracted from massive logs can effectively identify anomalous operational windows, avoiding the overhead of real-time or hierarchical analysis until a baseline is proven.

## Dataset
The core dataset consists of:
- **`unpivoted_data.parquet`**: A 413-million row unpivoted telemetry log containing various aggregated statistics across 5-minute intervals spanning four months. 
- **`anomaly_windows.csv`**: A list of 25 documented ground-truth anomaly windows.

**Important Note:** The true meaning of the `aggregated_stats_value` is not officially documented (e.g., whether it measures bytes, processing duration, or latency). Therefore, we treat these metrics as generalized statistical volumes.

## Why PySpark is Used
The raw dataset contains over 413 million rows, which far exceeds the memory capacity of typical single-machine DataFrames (e.g., Pandas). PySpark is used to perform distributed, out-of-core aggregation. By using PySpark's SQL engine, we effectively compress the 413-million row granular dataset into a compact ~39,000 row system-level feature table without overwhelming system memory.

## Feature Engineering
The pipeline uses PySpark to pivot and aggregate the data into **System-Wide 5-Minute Intervals**. 
The extracted features include:
- `total_request_volume`: Sum of all `count` stats.
- `client_request_volume`: Sum of `count` stats where kind is CLIENT.
- `server_request_volume`: Sum of `count` stats where kind is SERVER.
- `error_request_volume`: Sum of `count` stats where the status code is >= 500.
- `active_hosts`: Count of distinct hosts.
- `active_endpoints`: Count of distinct endpoints.
- `mean_recorded_avg_latency`: The arithmetic mean of all `avg` stats for the interval.

Raw statistical distributions (`min`, `max`, `std`, `skewness`) were excluded from the baseline model because they are heavily skewed by single outliers and cannot be naively averaged across conflicting dimensions. Before modeling, highly skewed volume metrics are transformed using `log1p`.

## Anomaly Detection Method
We use **Isolation Forest (Scikit-Learn)** as the baseline anomaly detector. 
Because Spark already handled the heavy lifting of feature engineering, the ML model operates entirely on the compact feature table in Pandas. We run multiple threshold experiments (`contamination = 0.01`, `0.025`, `0.05`) to avoid blindly asserting a single threshold.

## Evaluation
Evaluation is conducted at two levels:
1. **Interval-Level:** Precision, Recall, and F1-score are calculated for every discrete 5-minute tick compared to the labeled dataset.
2. **Window-Level:** Detected intervals are matched against the 25 known macroscopic anomaly windows (with a 5-minute boundary tolerance) to determine if the event as a whole was successfully flagged.

## How to Run the Project
Ensure you have Apache Spark installed (or the `pyspark` Python package) and run the pipeline script:
```bash
python pipeline.py
```
This script will:
1. Read the raw parquet file.
2. Perform distributed aggregations.
3. Save the engineered features to `outputs/system_features.parquet`.
4. Train the Isolation Forest.
5. Generate visualizations and metric reports in the `outputs/` folder.

## Dependencies
- `pyspark`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Limitations
- **Resolution:** This model operates strictly at a 5-minute global system level. It will detect macroscopic system failures but may mask localized single-endpoint failures.
- **Labels:** The ground-truth labels are merely used for evaluation. This is not a supervised classifier.
- **Missing Telemetry:** Some very short anomaly windows (e.g., 0 minutes long) fell entirely between our 5-minute telemetry ticks. These cannot be detected directly by this dataset.
- **Not Cyberattack Detection:** These anomalies represent statistical deviations in volume and latency. They do not definitively prove the presence of a cyberattack.
