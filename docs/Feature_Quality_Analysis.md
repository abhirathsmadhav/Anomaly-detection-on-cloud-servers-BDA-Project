# Feature Quality Analysis

Based on the provisional architecture (Primary observation unit: `interval_start, host, endpoint`) and the DuckDB analysis of the Parquet dataset, here are the results of the feature quality assessment.

## 1. Basic Observation Statistics

For the pivoted observations:
- **Total number of observations:** 10,415,124
- **Number of unique host-endpoint pairs:** 1,073
- **Average observations per host-endpoint pair:** ~9,706.5
- **Minimum observations per pair:** 1
- **Maximum observations per pair:** 39,365

## 2. Numerical Distributions & Outlier Detection

An analysis of the 8 aggregated numerical statistics reveals significant extreme outliers and some data quality issues that will heavily impact anomaly detection algorithms if used raw.

| Feature | Min | 1st pctl | 25th pctl | Median | 75th pctl | 99th pctl | Max | % Zero | % Negative |
|---------|-----|----------|-----------|--------|-----------|-----------|-----|--------|------------|
| **min** | 506.0 | 3,367.0 | 85,437.0 | 272,169.0 | 859,673.0 | 1.31e+07 | 9.00e+09 | 0.0% | 0.0% |
| **max** | 1,151.0 | 17,452.0 | 387,761.0 | 1,134,217.0 | 3,069,763.0 | 4.09e+07 | 9.09e+09 | 0.0% | 0.0% |
| **avg** | 1,151.0 | 6,901.6 | 178,231.2 | 534,511.9 | 1,365,360.8 | 1.88e+07 | 9.00e+09 | 0.0% | 0.0% |
| **median** | 1,151.0 | 5,832.0 | 137,989.3 | 418,457.0 | 1,196,478.2 | 2.06e+07 | 9.06e+09 | 0.0% | 0.0% |
| **count** | 1.0 | 1.0 | 4.0 | 12.0 | 54.0 | 1,568.0 | 48,823.0 | 0.0% | 0.0% |
| **std** | -50.0 | -50.0 | 47,736.2 | 192,435.3 | 698,165.0 | 1.03e+07 | 5.48e+09 | ~0.0% | 4.6% |
| **skewness**| -166.9 | -166.9 | 0.79 | 1.89 | 3.28 | 14.43 | 88.73 | ~0.0% | 17.6% |
| **kurtosis**| -60.0 | -60.0 | -1.17 | 3.83 | 12.45 | 249.33 | 7,879.97 | 0.0% | 29.6% |

### Outliers and Data Quality Flags
- **Extreme Right Tails:** Every latency metric (`min`, `max`, `avg`, `median`) and `count` possesses extreme right-tail outliers. For instance, the 99th percentile of `max` is ~40.9 million, but the absolute maximum is ~9.09 billion. Raw inclusion will severely compress the variance for inlier data in distance-based algorithms.
- **Impossible Values:** 
  - `std` contains negative values (-50.0 for 4.6% of the data), which is mathematically impossible. This is likely an artifact/placeholder for single-observation groups where standard deviation is undefined.
  - `kurtosis` has a minimum of -60.0. Excess kurtosis mathematically cannot fall below -2. This indicates another placeholder or computation artifact.

## 3. Correlations

An examination of the numerical features reveals high collinearity between specific metrics:
- **`avg` vs `median`:** 0.988 (Near perfect correlation)
- **`avg` vs `max`:** 0.954 (Very high correlation)
- **`std` vs `max`:** 0.842 (High correlation)
- **`count` vs `avg`:** 0.022 (No linear correlation)

## 4. Categorical / Context Fields

The categorical fields all exhibit manageable, low cardinality:
- **`kind` (CLIENT/SERVER):** 2 unique values.
- **`location`:** 7 unique values (e.g., datacenter4, datacenter6).
- **`method`:** 7 unique values (GET, POST, PUT, etc.).
- **`statusCode`:** 29 unique values (200, 403, 404, 401, etc.).

*Note: Since there are thousands of unique `endpoint` values, one-hot encoding them directly is not recommended, but these lower-cardinality fields can provide excellent context.*

## 5. Global Context Features

The global context features derived per `interval_start` (total request volume, client volume, server volume, average latency) show **very weak correlation** with the local host-endpoint features:
- `count` vs `total_req_vol`: 0.109
- `count` vs `client_vol`: 0.106
- `count` vs `server_vol`: 0.107
- `avg` vs `avg_latency`: 0.061

This is an excellent finding. It means the system-wide context features provide truly orthogonal information about the system's macro-state compared to the localized micro-state.

---

# Final Feature Recommendations

Based on the analysis, here is the recommended feature set for the anomaly detection model:

### A. Strong Candidate Numerical Features
1. **`stat_count` (Local Volume):** Highly uncorrelated with other features, crucial for volumetric anomalies.
2. **`stat_median` (Local Latency Central Tendency):** Preferred over `avg` as it is slightly more robust to the extreme right-tail values observed.
3. **`stat_max` (Local Peak Latency):** Good for catching instantaneous latency spikes.
4. **Global Context Features (`total_req_vol`, `client_vol`, `server_vol`, `avg_latency`):** These provide highly uncorrelated, orthogonal views of the macro system state.

### B. Features Requiring Transformation / Scaling
*None of the numerical features can be used in their raw format.*
- **Log Transformation / Robust Scaling:** `stat_count`, `stat_median`, `stat_max`, and all Global Features require transformations (e.g., `log1p` or `RobustScaler`) to mitigate the extreme multi-billion max-value outliers that would otherwise dominate distance and density calculations.
- **Imputation / Cleaning:** If `stat_std` is to be used, the negative placeholder values (`-50`) must be imputed (e.g., replaced with `0` or `NaN`).

### C. Features Best Kept as Contextual / Grouping Variables
- **`host` & `endpoint`:** Exclude from direct modeling features due to high cardinality. Use them strictly to identify the *location* of the anomaly post-detection.
- **`kind`, `location`, `method`:** Excellent candidates for grouping variables. You could either one-hot encode them (due to low cardinality) OR use them to train stratified/independent models (e.g., one model for CLIENT, one for SERVER).
- **`statusCode`:** Can be treated as a contextual feature or one-hot encoded, though its cardinality of 29 means stratification might spread data too thin.

### D. Features That Should Be Excluded
- **`stat_avg`:** Exclude to avoid extreme multicollinearity, as it has a 0.988 correlation with `stat_median`.
- **`stat_skewness` & `stat_kurtosis`:** Exclude for the baseline model. They contain impossible placeholder values (e.g., -60 for kurtosis) and are historically noisy and harder to interpret.
- **`stat_min`:** Usually stable and less informative for performance degradation anomalies compared to median/max.
